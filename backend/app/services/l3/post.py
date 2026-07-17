"""
Cuts v3 post-compute: deterministic assembly of the final ``cut_records``
from pass 2's judged output. No model call here, and no fallback -- the
remaining invariants (zero overlap, boundary-on-edge) are enforced in code; a
violation fails the ingest run loudly for re-run rather than being silently
patched over. Full coverage is NO LONGER an invariant: cuts are a selection,
not a partition, so gaps (dropped connective tissue) are legal. See
cuts_v3.plan.md section 6 and cuts_v3_boundaries_v2.plan.md.

Note on "framing motion" (plan sec. 6, the subject-centroid-follows-crop
bullet): that machinery already exists in ``app.services.l3.framing``
(``focus_for_range``), reading straight off ``motion_dynamics`` for an
arbitrary ``(file_id, src_in_ms, src_out_ms)`` span at ARRANGE time. A
cut_record's own ``file_id``/``src_in_ms``/``src_out_ms`` are already
everything that machinery needs, so there is nothing new to build or store
here for that bullet.
"""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import Lattice, _anchors_in, resolve_speech_span_ms
from app.services.l3.pass1 import JunkSuspect
from app.services.l3.pass2 import Pass2Cut, Pass2Output
from app.services.l3.post_params import (
    ANCHOR_PAD_MS, CAMERA_FOLLOW_ACTION, CAMERA_FOLLOW_COHERENCE, CAMERA_PAN_RATE,
    CAMERA_SHAKE_COHERENCE, CAMERA_SHAKE_RATE, CAMERA_ZOOM_RATE, ENERGY_GRADE_BANDS,
    FLATLINE_BAND, PACE_LEVEL_TARGETS, SPEED_CEIL, SPEED_FLOOR,
    V4_MIN_MS_DENSE_BONUS, V4_MIN_MS_FLOOR,
)
from app.services.l3.seam import BREAK_BOUNDARY_REASONS, Seam, classify_seam
from app.services.l3.sync import av_couple
from app.services.l3.video_segments import _sharpest_ms

logger = logging.getLogger(__name__)


@dataclass
class PaceEnvelope:
    min_ms: int
    natural_ms: int
    max_ms: int
    levels: List[float]
    energy_grade: str
    natural_sound: bool
    # Removable dead-air + filler spans across a SPEECH cut (absolute ms, inside
    # [src_in, src_out]) that the dial MAY shave -- edge silence/fillers, interior
    # disfluencies, and pause-excess. Code owns these numbers; the dial (view-math)
    # owns how much of them to apply. Empty for video or a clean spoken beat.
    remove_spans: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"min_ms": self.min_ms, "natural_ms": self.natural_ms, "max_ms": self.max_ms,
                "levels": list(self.levels), "energy_grade": self.energy_grade,
                "natural_sound": self.natural_sound,
                "remove_spans": [list(sp) for sp in self.remove_spans]}


# Fillers safe to shave from the EDGES of a spoken beat (a leading "so"/"you
# know" or trailing "right" is throat-clearing). Interior removal uses the
# tighter _INTERIOR_FILLER_TOKENS below -- a mid-line "so"/"like"/"right" is
# usually real content, so we never touch it. Both sets + how the dial scales
# the budget are the whole tuning knob.
_FILLER_EDGE_TOKENS = {
    "um", "uh", "umm", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mmm", "uhh",
    "so", "well", "okay", "ok", "like", "right", "yeah", "anyway", "basically",
    "actually", "literally", "you", "know", "i", "mean",
}
_INTERIOR_FILLER_TOKENS = {
    "um", "uh", "umm", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mmm", "uhh",
}
_FILLER_WORD_RE = re.compile(r"[a-z']+")


def _pure_filler(text: Optional[str], vocab: set) -> bool:
    """True only if EVERY alphabetic token in the word is in ``vocab`` (so 'um,'
    counts, but 'important' never does)."""
    toks = _FILLER_WORD_RE.findall((text or "").lower())
    return bool(toks) and all(t in vocab for t in toks)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort + merge overlapping/adjacent [start, end] spans (drops empties)."""
    ordered = sorted((int(a), int(b)) for a, b in spans if int(b) > int(a))
    out: List[Tuple[int, int]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def compute_speech_remove_spans(
    words: List[dict], word_span: Optional[Tuple[int, int]], s: int, e: int
) -> List[Tuple[int, int]]:
    """Deterministic removable dead-air + filler spans across a speech cut, for
    the dial to shave (edge AND interior). Everything is derived from the cut's
    OWN word timings -- no absolute constants -- so nothing "real" is ever
    proposed for removal:
      * EDGE silence + edge fillers: the run of pure-filler words at each end
        plus the dead air out to the cut boundary.
      * INTERIOR fillers: clear disfluencies ('um'/'uh'/...) mid-line only.
      * INTERIOR pauses: silence between kept words BEYOND the speaker's own
        median inter-word gap (their natural rhythm); only the EXCESS is
        removable, taken from the middle so a natural beat of silence remains.
    Returns a sorted, merged list of [start_ms, end_ms] inside [s, e]. The dial
    (view-math) decides how much of this budget actually gets cut."""
    if not words or word_span is None:
        return []
    a, b = int(word_span[0]), int(word_span[1])
    a, b = max(0, a), min(len(words) - 1, b)
    if a > b:
        return []
    content = [i for i in range(a, b + 1) if not _pure_filler(words[i].get("text"), _FILLER_EDGE_TOKENS)]
    if not content:
        return []  # an all-filler span is left whole (likely junk, handled elsewhere)
    first, last = content[0], content[-1]
    spans: List[Tuple[int, int]] = []

    # Edges: dead air + filler run out to each boundary.
    cs = int(words[first].get("start_ms", s))
    ce = int(words[last].get("end_ms", e))
    if cs > s:
        spans.append((s, cs))
    if e > ce:
        spans.append((ce, e))

    # Interior: fillers -> remove the word; everything else is "kept" and sets
    # the natural-rhythm baseline for pause trimming.
    kept: List[int] = []
    for i in range(first, last + 1):
        if i not in (first, last) and _pure_filler(words[i].get("text"), _INTERIOR_FILLER_TOKENS):
            ws, we = int(words[i].get("start_ms")), int(words[i].get("end_ms"))
            if we > ws:
                spans.append((ws, we))
        else:
            kept.append(i)

    # Interior pauses: excess over the speaker's median gap between kept words.
    gaps = [(int(words[kept[j]].get("end_ms")), int(words[kept[j + 1]].get("start_ms")))
            for j in range(len(kept) - 1)]
    positive = [g1 - g0 for g0, g1 in gaps if g1 - g0 > 0]
    if positive:
        baseline = statistics.median(positive)
        for g0, g1 in gaps:
            excess = (g1 - g0) - baseline
            if excess > 0:
                keep = baseline / 2.0  # leave a natural beat on each side
                rs, re_ = int(round(g0 + keep)), int(round(g1 - keep))
                if re_ > rs:
                    spans.append((rs, re_))

    return _merge_spans(spans)


# --------------------------------------------------------------------------
# Quality scores -- two deterministic 0..1 numbers stamped on every cut (no
# LLM number, no magic threshold on a raw measurement):
#   * speech_quality -- delivery ONLY (crispness + loudness). Camera-
#     independent: the same words spoken once score the same regardless of
#     which simultaneous angle filmed them. None for a cut with no speech.
#   * total_quality  -- speech (if any) blended with visual presentation
#     (on-camera, shot tightness, sharpness, look). Biases toward on-camera
#     close-ups; this is the number that crowns the winner WITHIN a same-
#     setting take cluster (never across outlook angles -- see
#     _enforce_take_winner).
# Every continuous term is normalised against the CLIP's OWN min/max so there
# are no absolute dB/pixel constants; the one category->rank map (_SHOT_TIGHTNESS)
# is a fixed, interpretable ordinal scale, not a tuned threshold.
# --------------------------------------------------------------------------

_SHOT_TIGHTNESS = {
    "extreme_close_up": 1.0, "close_up": 0.9, "medium_close_up": 0.75,
    "medium": 0.6, "medium_wide": 0.45, "wide": 0.3, "extreme_wide": 0.15,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _series_lohi(arr: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Min/max of a signal series (ignoring None), for clip-relative
    normalisation. (None, None) when there's nothing usable."""
    vals = [float(v) for v in (arr or []) if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def _norm_in_clip(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    if value is None or lo is None or hi is None or hi <= lo:
        return None
    return _clamp01((value - lo) / (hi - lo))


def compute_speech_quality(
    rms_db: List[float], hop_ms: int, s: int, e: int, remove_budget_ms: int,
    rms_lo: Optional[float], rms_hi: Optional[float],
) -> float:
    """Delivery quality of a spoken beat, 0..1: how much of it is clean speech
    (fluency = 1 - removable dead-air/filler fraction) blended with how present
    the voice is (loudness, normalised against the clip's own rms range).
    Camera-independent by construction -- fluency comes from word timings and
    loudness from the source audio, both shared by simultaneous angles."""
    dur = max(1, e - s)
    fluency = _clamp01(1.0 - remove_budget_ms / dur)
    terms = [fluency]
    loudness = _norm_in_clip(_mean_in_span(rms_db, hop_ms, s, e), rms_lo, rms_hi) if rms_db else None
    if loudness is not None:
        terms.append(loudness)
    return round(_mean(terms) or 0.0, 3)


# perception_upgrade.plan.md Part C2 (optional downstream): a technically
# poor shot_quality nudges total_quality DOWN -- gated on the field being
# PRESENT (a bad/unknown value from an old, pre-migration run is simply
# absent from `framing`, so `.get()` returns None and this is a no-op; old
# runs are unaffected by construction, never explicitly branched on).
# "stable"/"unsure" are deliberately absent (no penalty -- unsure means the
# model genuinely couldn't tell, not that the shot is bad).
_SHOT_QUALITY_PENALTY = {
    "shaky": 0.7, "whip": 0.7, "soft_focus": 0.6, "racking_focus": 0.8, "exposure_shift": 0.8,
}


def compute_visual_score(
    on_camera: Optional[bool], framing: Dict[str, Any], look: Dict[str, Any],
    blur: List[float], hop_ms: int, s: int, e: int,
    blur_lo: Optional[float], blur_hi: Optional[float],
) -> Optional[float]:
    """Presentation quality of the pixels, 0..1 (mean of whatever terms are
    available): subject on camera, framing tightness (shot_size ordinal),
    sharpness (clip-relative, since blur is unitless), and a clean/graded look.
    None when NOTHING visual is known (e.g. a bare speech cut with no framing).
    A technically poor shot_quality (present only on runs that have it) then
    scales the result down -- see _SHOT_QUALITY_PENALTY."""
    terms: List[float] = []
    if on_camera is not None:
        terms.append(1.0 if on_camera else 0.0)
    tight = _SHOT_TIGHTNESS.get((framing or {}).get("shot_size"))
    if tight is not None:
        terms.append(tight)
    blur_norm = _norm_in_clip(_mean_in_span(blur, hop_ms, s, e), blur_lo, blur_hi) if blur else None
    if blur_norm is not None:
        terms.append(1.0 - blur_norm)  # least-blurred in the clip -> sharpest -> 1.0
    if look:
        clean = not (look.get("exposure_flags") or [])
        terms.append(1.0 if (bool(look.get("graded")) and clean) else (0.7 if clean else 0.4))
    result = _mean(terms)
    if result is None:
        return None
    penalty = _SHOT_QUALITY_PENALTY.get((framing or {}).get("shot_quality"))
    return result * penalty if penalty is not None else result


def compute_total_quality(kind: str, speech_quality: Optional[float],
                          visual_score: Optional[float]) -> float:
    """The single rank number for take selection + arrangement, 0..1. Speech
    cuts blend delivery and presentation equally (so among identical-audio
    angles the on-camera close-up wins); video cuts are visual-only."""
    if kind == "speech":
        terms = [t for t in (speech_quality, visual_score) if t is not None]
        return round(_mean(terms) or 0.0, 3)
    return round(visual_score if visual_score is not None else 0.0, 3)


# --------------------------------------------------------------------------
# Camera-move label: one plain phrase per cut the brain can read directly
# (static | pan left | pan right | tilt up | tilt down | zoom in | zoom out |
#  follow subject | shaky | unknown). Deterministic, from the SIGNED camera
# velocity L1 already fits per hop -- no model call, no per-frame guessing.
# --------------------------------------------------------------------------

def _span_slice(arr: List[float], hop_ms: int, s: int, e: int) -> List[float]:
    if not arr or hop_ms <= 0:
        return []
    lo = max(0, s // hop_ms)
    hi = min(len(arr) - 1, max(lo, (e - 1) // hop_ms))
    return arr[lo:hi + 1]


def classify_camera_move(motion: Dict[str, Any], s: int, e: int) -> str:
    """The camera's behaviour over ``[s, e)`` as a single phrase. Reads the
    signed per-hop velocity (camera_dx/dy = frame-fraction travel, camera_zoom =
    scale change) plus coherence/action to name the dominant move. Sign
    convention (from scene flow): +dx = camera pans LEFT, +dy = tilts UP,
    +zoom = zooms IN. 'unknown' when there's no motion signal for the span."""
    hop_ms = int(motion.get("hop_ms") or 0)
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    if not dx and not dy and not dz:
        return "unknown"
    dur_s = max(1e-3, (e - s) / 1000.0)

    # Net (signed) travel per second along each axis -> is this a real move?
    rate_x = sum(dx) / dur_s
    rate_y = sum(dy) / dur_s
    rate_z = sum(dz) / dur_s

    # Strength of each candidate, normalised to its own threshold so the axes
    # are comparable; the biggest wins.
    cands = {
        "zoom": abs(rate_z) / CAMERA_ZOOM_RATE,
        "pan": abs(rate_x) / CAMERA_PAN_RATE,
        "tilt": abs(rate_y) / CAMERA_PAN_RATE,
    }
    axis, strength = max(cands.items(), key=lambda kv: kv[1])

    if strength < 1.0:
        # No directed move. Agitated frame with a near-zero net path + a model
        # that won't hold = hand-held shake; otherwise a locked-off static shot.
        jitter = (sum(abs(v) for v in dx) + sum(abs(v) for v in dy)) / dur_s
        coh = _mean(_span_slice(motion.get("camera_coherence") or [], hop_ms, s, e)) or 1.0
        if jitter >= CAMERA_SHAKE_RATE and coh < CAMERA_SHAKE_COHERENCE:
            return "shaky"
        return "static"

    if axis == "zoom":
        return "zoom in" if rate_z > 0 else "zoom out"

    # A translation move that also tracks a busy subject in a coherent frame is
    # the camera FOLLOWING, not a free pan across a scene.
    action = _mean(_span_slice(motion.get("action_energy") or [], hop_ms, s, e)) or 0.0
    coh = _mean(_span_slice(motion.get("camera_coherence") or [], hop_ms, s, e)) or 0.0
    if action >= CAMERA_FOLLOW_ACTION and coh >= CAMERA_FOLLOW_COHERENCE:
        return "follow subject"
    if axis == "pan":
        return "pan left" if rate_x > 0 else "pan right"
    return "tilt up" if rate_y > 0 else "tilt down"


# --------------------------------------------------------------------------
# Salience (perception_upgrade.plan.md Part D, "F8"): the cut's single
# strongest INSTANT, fused from signals L1 already computed. Deterministic,
# code-owned -- NOT the LLM's job (it's a number), and DISTINCT from
# hero_ts_ms (the best still for DISPLAY; salience is the strongest EVENT
# moment -- useful for emphasis, thumbnail choice, punch-in timing, and for
# the brain to know where a cut peaks).
# --------------------------------------------------------------------------

def _salience(
    action_energy: List[float], hop_ms: int, s: int, e: int,
    ae_lo: Optional[float], ae_hi: Optional[float],
    rms_db: List[float], rms_hop_ms: int, rms_lo: Optional[float], rms_hi: Optional[float],
    anchors: List[int], onsets_ms: List[int], hero_ts_ms: int,
) -> Dict[str, Any]:
    """A per-hop curve over [s, e) fusing (a) normalized action_energy,
    (b) normalized loudness, and (c) a flat bump at any onset/anchor instant
    inside the span (a moment L1 already flagged as "something happened
    here") -- reusing the SAME clip-relative normalization
    (_norm_in_clip/_series_lohi) the quality scores use, so there are no
    absolute energy/dB constants. peak_ms = argmax (absolute ms); score =
    the peak's height normalized against the CUT's own curve range, 0..1.
    No usable signal at all -> {peak_ms: hero_ts_ms, score: 0.0} (the
    already-computed best STILL, never a fabricated peak)."""
    no_signal = {"peak_ms": hero_ts_ms, "score": 0.0}
    if hop_ms <= 0 or e <= s:
        return no_signal
    lo_i, hi_i = s // hop_ms, max(s // hop_ms, (e - 1) // hop_ms)
    n = hi_i - lo_i + 1
    if n <= 0:
        return no_signal

    curve = [0.0] * n
    have_signal = False

    for i in range(n):
        ae_i = lo_i + i
        if ae_i < len(action_energy):
            v = _norm_in_clip(action_energy[ae_i], ae_lo, ae_hi)
            if v is not None:
                curve[i] += v
                have_signal = True

    if rms_db and rms_hop_ms > 0:
        for i in range(n):
            bin_ms = (lo_i + i) * hop_ms
            rms_i = bin_ms // rms_hop_ms
            if 0 <= rms_i < len(rms_db):
                v = _norm_in_clip(rms_db[rms_i], rms_lo, rms_hi)
                if v is not None:
                    curve[i] += v
                    have_signal = True

    for t in list(anchors or []) + [t for t in (onsets_ms or []) if s <= t < e]:
        i = (int(t) // hop_ms) - lo_i
        if 0 <= i < n:
            curve[i] += 1.0
            have_signal = True

    if not have_signal:
        return no_signal

    peak_i = max(range(n), key=lambda i: curve[i])
    peak_val = curve[peak_i]
    c_lo, c_hi = min(curve), max(curve)
    score = _clamp01((peak_val - c_lo) / (c_hi - c_lo)) if c_hi > c_lo else (1.0 if peak_val > 0 else 0.0)
    return {"peak_ms": int((lo_i + peak_i) * hop_ms), "score": round(score, 3)}


@dataclass
class CutRecord:
    file_id: str
    src_in_ms: int
    src_out_ms: int
    kind: str                              # "speech" | "video"
    word_span: Optional[Tuple[int, int]]
    atom_ids: Optional[List[int]]
    label: str
    summary: str
    on_camera: Optional[bool]
    junk: bool
    junk_reason: str
    framing: Dict[str, Any]
    look: Dict[str, Any]
    caption_zones: List[Tuple[float, float, float, float]]
    hero_ts_ms: int
    pace: PaceEnvelope
    take_group_id: Optional[str]
    take_role: Optional[str]
    channel: str                           # "said" | "done" | "shown"
    # Two deterministic 0..1 quality scores (see the scoring section above).
    # speech_quality is None for a cut with no speech; total_quality is always
    # set and is what crowns a same-setting take group's winner.
    speech_quality: Optional[float]
    total_quality: float
    # Per-person appearance fingerprints from pass 2 (list of {description,
    # position, appearance}) -- lets take/outlook grouping + "show the speaker"
    # arrange logic recognise the same person across cuts by eye.
    characteristics: List[Dict[str, Any]] = field(default_factory=list)
    # Deterministic per-cut continuity (cuts_v3_continuity.plan.md): this cut's
    # 1-based ordinal + total among ALL cuts on its clip (incl. junk -- a gap in
    # the numbering IS the signal a junk beat sits there) and whether each
    # neighbor is a weldable continuation (seam.classify_seam) or a hard cut.
    # {clip, cut_no, of, prev_contiguous, next_contiguous, seam_reason_prev,
    #  seam_reason_next}. Computed ONCE here (the ingest signals are richest);
    # read paths (brain + UI) just read the block.
    continuity: Dict[str, Any] = field(default_factory=dict)
    # A single plain-language camera-move phrase for the span (static, pan
    # left/right, tilt up/down, zoom in/out, follow subject, shaky, unknown) --
    # so the brain knows how a shot moves without reading raw motion signals.
    camera: str = "unknown"
    # audio_sync.plan.md SS6 "pinning": which sync_groups row (if any) this
    # cut's file belonged to AT INGEST TIME. None for a non-synced file
    # (the overwhelming common case). Snapshotting this (rather than joining
    # sync_group_members live) means a later re-sync of the same files can't
    # retroactively change what an already-ingested edit's audio came from.
    sync_group_id: Optional[str] = None
    # perception_upgrade.plan.md Part C3: on-screen text/graphics (slide,
    # lower-third, UI, title) the model read off the pixels -- "" when none.
    # LLM-owned free text; unlocks tutorials/explainers/screen-recordings.
    screen_text: str = ""
    # perception_upgrade.plan.md Part D ("F8"): the cut's single strongest
    # INSTANT, code-computed -- see _salience. {} on a cut with no signal.
    salience: Dict[str, Any] = field(default_factory=dict)
    # voice_first_identity.plan.md Phase C/D/G -- all three code-derived,
    # never LLM-echoed:
    #   - voice_ids: global voice(s) heard in this cut (Pass 1 word-level
    #     diarization, mapped through voice clustering). [] for a video cut.
    #   - speaker_person: the global person id (Px) the speaker pass bound
    #     the speaking voice to. None = honest "owner unknown", never a guess.
    #   - visible_persons: every global person id visible on screen in this
    #     cut (per-cut-occurrence face clustering) -- a cut can show several.
    voice_ids: List[str] = field(default_factory=list)
    speaker_person: Optional[str] = None
    visible_persons: List[str] = field(default_factory=list)
    # av_coupling_authoritative.plan.md: this cut's AUTHORITATIVE audio,
    # baked at assembly time -- a coupled (video, audio) unit, never
    # re-derived lazily at render time. audio_file_id == file_id, offset 0
    # for the ~90% common case (no sync group, or this file already IS the
    # group's authoritative source); otherwise the group's authoritative
    # file + a per-cut REFINED offset (sync.av_couple.refine_offset,
    # cross-correlated against this cut's own audio window -- never just
    # the group's loose global delta, which is what drifted into visible
    # lip-sync error). audio_align_confidence is None for a same-source cut
    # or when the refinement's guard rejected a weak/ambiguous peak and fell
    # back to the unrefined global delta.
    audio_file_id: str = ""
    audio_offset_ms: int = 0
    audio_align_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id, "src_in_ms": self.src_in_ms, "src_out_ms": self.src_out_ms,
            "kind": self.kind,
            "word_span": list(self.word_span) if self.word_span else None,
            "atom_ids": self.atom_ids, "label": self.label, "summary": self.summary,
            "on_camera": self.on_camera,
            "junk": self.junk, "junk_reason": self.junk_reason,
            "framing": self.framing, "look": self.look,
            "caption_zones": [list(z) for z in self.caption_zones],
            "hero_ts_ms": self.hero_ts_ms, "pace": self.pace.to_dict(),
            "take_group_id": self.take_group_id, "take_role": self.take_role,
            "channel": self.channel, "continuity": self.continuity,
            "speech_quality": self.speech_quality, "total_quality": self.total_quality,
            "characteristics": self.characteristics, "camera": self.camera,
            "sync_group_id": self.sync_group_id,
            "screen_text": self.screen_text, "salience": self.salience,
            "voice_ids": self.voice_ids, "speaker_person": self.speaker_person,
            "visible_persons": self.visible_persons,
            "audio_file_id": self.audio_file_id, "audio_offset_ms": self.audio_offset_ms,
            "audio_align_confidence": self.audio_align_confidence,
        }


# --------------------------------------------------------------------------
# hero_ts_ms: anchor > subject-sharp > midpoint
# --------------------------------------------------------------------------

def pick_hero_ts_ms(anchors: List[int], blur: List[float], hop_ms: int, s: int, e: int) -> int:
    if anchors:
        return anchors[0]
    return _sharpest_ms(blur, hop_ms, s, e, (s + e) // 2)


# --------------------------------------------------------------------------
# Pace envelope
# --------------------------------------------------------------------------

def _mean_in_span(arr: List[float], hop_ms: int, s: int, e: int) -> float:
    if not arr or hop_ms <= 0:
        return 0.0
    lo, hi = max(0, s // hop_ms), min(len(arr) - 1, max(s // hop_ms, (e - 1) // hop_ms))
    if hi < lo:
        return 0.0
    seg = arr[lo:hi + 1]
    return sum(seg) / len(seg) if seg else 0.0


def _flatline_bound_ms(action_energy: List[float], hop_ms: int, from_ms: int, ceiling_ms: int) -> int:
    """How far past ``from_ms`` action_energy stays within FLATLINE_BAND of
    its own value at ``from_ms``, capped at ``ceiling_ms`` (the next cut's
    start, or the file's end) -- past that point there's nothing new
    happening to justify a longer hold."""
    if not action_energy or hop_ms <= 0 or ceiling_ms <= from_ms:
        return from_ms
    i0 = min(len(action_energy) - 1, from_ms // hop_ms)
    baseline = action_energy[i0]
    ceiling_i = ceiling_ms // hop_ms
    i = i0
    while i + 1 < len(action_energy) and i < ceiling_i:
        i += 1
        if abs(action_energy[i] - baseline) > FLATLINE_BAND:
            return min(i * hop_ms, ceiling_ms)
    return ceiling_ms


def _anchor_span_ms(anchors: List[int]) -> int:
    if not anchors:
        return 0
    return (max(anchors) - min(anchors)) + 2 * ANCHOR_PAD_MS


def _energy_grade(mean_action_energy: float) -> str:
    for grade, upper in ENERGY_GRADE_BANDS:
        if mean_action_energy < upper:
            return grade
    return "high"


def _pace_levels(intrinsic_velocity: float, min_speed: float, max_speed: float) -> List[float]:
    lo, hi = max(min_speed, SPEED_FLOOR), min(max_speed, SPEED_CEIL)
    if lo > hi:
        lo, hi = hi, lo   # an inverted taste fence -- clamp to a single reachable point rather than crash
    if intrinsic_velocity <= 0:
        return [hi] * len(PACE_LEVEL_TARGETS)
    return [min(max(target / intrinsic_velocity, lo), hi) for target in PACE_LEVEL_TARGETS]


def compute_pace_envelope(
    *, kind: str, s: int, e: int, readability_ms: int, anchors: List[int],
    action_energy: List[float], hop_ms: int,
    next_cut_start_ms: int, max_tasteful_speed: float, min_tasteful_speed: float,
    natural_sound: bool, density: Optional[float] = None,
) -> PaceEnvelope:
    natural_ms = max(1, e - s)
    mean_ae = _mean_in_span(action_energy, hop_ms, s, e)
    grade = _energy_grade(mean_ae)

    if kind == "speech":
        # Speech always plays at native speed -- see cuts_v3.plan.md sec. 6.
        return PaceEnvelope(min_ms=max(readability_ms, natural_ms), natural_ms=natural_ms,
                            max_ms=natural_ms, levels=[1.0] * len(PACE_LEVEL_TARGETS),
                            energy_grade=grade, natural_sound=natural_sound)

    # min_ms floors tightening at readability + either the anchor envelope
    # (V3 -- impacts stay in frame) or, for a V4 cut (density given), the
    # segmenter's own event density: a sparse/monotonous span collapses hard
    # at high energy, a dense one holds more room so real events aren't
    # clipped (cuts_v4_segmentation.plan.md section 6). No camera-move floor
    # either way: the derived pan label is gone (deterministic-keep), and
    # pace stays purely signal-driven.
    if density is not None:
        min_ms = max(readability_ms, round(V4_MIN_MS_FLOOR + density * V4_MIN_MS_DENSE_BONUS))
    else:
        min_ms = max(readability_ms, _anchor_span_ms(anchors))
    flatline_end_ms = _flatline_bound_ms(action_energy, hop_ms, e, next_cut_start_ms)
    max_ms = max(natural_ms, flatline_end_ms - s)
    levels = _pace_levels(mean_ae, min_tasteful_speed, max_tasteful_speed)
    return PaceEnvelope(min_ms=min_ms, natural_ms=natural_ms, max_ms=max_ms, levels=levels,
                        energy_grade=grade, natural_sound=natural_sound)


# --------------------------------------------------------------------------
# Invariant enforcement: zero overlap, per file (coverage gaps are legal).
# --------------------------------------------------------------------------

def _validate_no_overlap(file_id: str, spans: List[Tuple[int, int]], duration_ms: int) -> None:
    """Boundaries-v2: cuts are a SELECTION, not a partition -- GAPS ARE LEGAL
    (connective tissue / pre-roll / dead air is dropped, not tiled). The only
    invariant left is zero overlap: two cuts must never claim the same instant,
    or the timeline is ambiguous. Coverage gaps used to raise here; that was the
    full-coverage invariant that forced every dead sliver to become a junk tile
    (see cuts_v3_boundaries_v2.plan.md)."""
    spans = sorted(spans)
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        if s1 < e0:
            raise ValueError(f"{file_id}: overlap between [{s0}-{e0}] and [{s1}-{e1}]")


# --------------------------------------------------------------------------
# Continuity (cuts_v3_continuity.plan.md): cut_no/of + weldable-neighbor flags,
# computed ONCE here (at ingest) from the same lattice/atom signals pass 1's
# own word-gap seam test reads (`pass1._gap_seam`) -- generalized from a
# word-to-word seam to a CUT-to-cut seam on the same clip.
# --------------------------------------------------------------------------

def _junk_suspect_spans(junk_suspects: List[JunkSuspect], lattice: Lattice) -> List[Tuple[int, int]]:
    """Every pass-1 junk suspect for one clip, resolved to its ms span (word or
    atom edges -- the same resolution ``assemble_cut_records`` uses for a real
    cut). Unresolvable suspects (a bad index) are skipped, not fabricated."""
    out: List[Tuple[int, int]] = []
    words = lattice.words
    atoms_by_id = {a.atom_id: a for a in lattice.atoms}
    for js in junk_suspects:
        if js.word_span is not None:
            a, b = js.word_span
            if 0 <= a <= b < len(words):
                out.append((int(words[a].get("start_ms", 0)), int(words[b].get("end_ms", 0))))
            continue
        if js.atom_ids:
            members = [atoms_by_id[i] for i in js.atom_ids if i in atoms_by_id]
            if members:
                out.append((min(m.start_ms for m in members), max(m.end_ms for m in members)))
    return out


def _has_scene_or_transition(atoms: List, gap_lo: int, gap_hi: int, *, synced: bool = False) -> bool:
    """A break-type atom boundary (shot cut / wipe / degenerate -- never an
    energy-regime edge) lands AT the seam's two boundary points (the common
    case: the cuts touch, ``gap_lo == gap_hi``) or strictly inside a nonzero
    gap between them. Mirrors ``pass1._gap_seam``'s break-edge test,
    including its ``synced`` skip (audio_sync.plan.md SS7.6): a synced
    multicam file's picture is decoupled from the speech beat, so its own
    shot boundary must not read as a hard continuity break."""
    if synced:
        return False
    for a in atoms:
        if a.end_ms == gap_lo and a.state_out in BREAK_BOUNDARY_REASONS:
            return True
        if a.start_ms == gap_hi and a.state_in in BREAK_BOUNDARY_REASONS:
            return True
    if gap_hi > gap_lo:
        for a in atoms:
            if ((a.state_in in BREAK_BOUNDARY_REASONS and gap_lo < a.start_ms < gap_hi)
                    or (a.state_out in BREAK_BOUNDARY_REASONS and gap_lo < a.end_ms < gap_hi)):
                return True
    return False


def _has_scene_or_transition_v4(motion: Optional[dict], gap_lo: int, gap_hi: int, *, synced: bool = False) -> bool:
    """v4_cuts_as_primitive.plan.md section 5: the V4 seam test, keyed off
    the CUTS' own boundaries rather than atom micro-edges (which a V4 cut's
    span generally doesn't align with -- atoms are no longer part of the
    video path at all). A nonzero gap between two adjacent V4 cuts means the
    segmenter deliberately left that stretch out of both -- discarded scrap,
    a real break. An edge-touching (zero-gap) seam is a hard break only if a
    genuine transition (wipe/degenerate -- a motion-level signal, independent
    of atoms) sits at the touch point; otherwise the two cuts are a
    continuous, deliberately-adjacent moment (e.g. a camera-move cut's
    settle immediately followed by the next one in the same shot)."""
    if synced:
        return False
    if gap_hi > gap_lo:
        return True
    for p in (motion or {}).get("transition_points") or []:
        ts = p.get("ts_ms")
        if p.get("kind") in ("wipe", "degenerate") and ts is not None and gap_lo <= int(ts) <= gap_hi:
            return True
    return False


def _has_flagged_break(junk_spans: List[Tuple[int, int]], gap_lo: int, gap_hi: int) -> bool:
    """A pass-1 junk suspect (production cue, false start, dead air) overlaps
    the gap -- or, for a zero-width gap (the cuts touch), contains the seam
    point exactly."""
    for s, e in junk_spans:
        if gap_hi > gap_lo:
            if max(s, gap_lo) < min(e, gap_hi):
                return True
        elif s <= gap_lo <= e:
            return True
    return False


def _clip_continuity(
    file_id: str, idxs: List[int],
    resolved: List[Tuple[Pass2Cut, int, int, List[int]]],
    lattice: Lattice, junk_spans: List[Tuple[int, int]],
    motion: Optional[dict] = None, v4_meta_by_ref: Optional[Dict[str, Dict[str, Any]]] = None,
    *, synced: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """cut_no/of/prev_contiguous/next_contiguous/seam_reason_* for every cut on
    one clip, in source order over ALL cuts (incl. junk). One seam verdict per
    adjacent pair fills BOTH sides (cur's next_contiguous, next's
    prev_contiguous) so the two always agree. ``idxs`` must already be sorted
    by src_in_ms. Returns ``{resolved_idx: continuity_dict}``.

    ``v4_meta_by_ref`` (v4_cuts_as_primitive.plan.md section 5): when BOTH
    cuts of a pair are V4 video cuts (their source_ref is a key in this dict),
    the seam test keys off the cuts' own boundaries instead of atoms -- see
    ``_has_scene_or_transition_v4``. Any pair involving a speech cut (or a V3
    video cut) keeps reading the atom lattice exactly as before -- speech
    boundaries are unaffected by V4. None/empty -> always the V3 atom path."""
    v4_meta_by_ref = v4_meta_by_ref or {}
    n = len(idxs)
    conts = [{
        "clip": file_id, "cut_no": pos + 1, "of": n,
        "prev_contiguous": False, "next_contiguous": False,
        "seam_reason_prev": None, "seam_reason_next": None,
    } for pos in range(n)]
    for pos in range(n - 1):
        cur_cut, cs, ce, _ = resolved[idxs[pos]]
        nxt_cut, ns, ne, _ = resolved[idxs[pos + 1]]
        both_v4 = cur_cut.source_ref in v4_meta_by_ref and nxt_cut.source_ref in v4_meta_by_ref
        has_break = (_has_scene_or_transition_v4(motion, ce, ns, synced=synced) if both_v4
                    else _has_scene_or_transition(lattice.atoms, ce, ns, synced=synced))
        verdict = classify_seam(Seam(
            same_clip=True,
            # voice_first_identity.plan.md: voice_ids is the deterministic
            # ground truth (Pass 1 word-level diarization + voice
            # clustering) -- unlike speaker_person it never depends on the
            # speaker pass's binding succeeding. Two video cuts (both []) or
            # two cuts sharing a voice both read as "same speaker" (a
            # permissive default -- other seam signals still decide).
            same_speaker=(set(cur_cut.voice_ids) == set(nxt_cut.voice_ids)),
            gap_ms=max(0, ns - ce),
            bridged_speech_ms=(ce - cs) + (ne - ns),
            has_scene_or_transition=has_break,
            has_flagged_break=_has_flagged_break(junk_spans, ce, ns),
        ))
        conts[pos]["next_contiguous"] = verdict.weldable
        conts[pos]["seam_reason_next"] = verdict.reason
        conts[pos + 1]["prev_contiguous"] = verdict.weldable
        conts[pos + 1]["seam_reason_prev"] = verdict.reason
    return {idxs[pos]: conts[pos] for pos in range(n)}


# --------------------------------------------------------------------------
# Assembly
#
# No action-protection override here any more (deterministic-keep): it used
# hardcoded thresholds (>=2 anchors, mean_energy>=0.45) to un-junk a span the
# model discarded -- exactly the band-aid the plan removes. Keeping a real
# action is now guaranteed upstream instead: pass 1's total-coverage fill
# means an action is never silently dropped, and junk is a recoverable label
# (shown in the Discarded tray), never a deletion. Code no longer second-
# guesses the model's semantic junk call with a number.
# --------------------------------------------------------------------------

def assemble_cut_records(
    pass2_output: Pass2Output,
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, Dict[str, Any]],
    silences_by_file: Dict[str, List[dict]],
    junk_suspects: Optional[List[JunkSuspect]] = None,
    audio_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    synced_file_ids: Optional[set] = None,
    sync_group_by_file: Optional[Dict[str, str]] = None,
    sync_info_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    v4_meta_by_ref: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[CutRecord]:
    """Resolve every judged cut to its final ms span (word/atom edges only,
    by construction), enforce zero-overlap per file (gaps are legal -- cuts are
    a selection, not a partition), then compute hero_ts_ms + the pace envelope
    + continuity (cut_no/of + weldable-neighbor flags, over ALL cuts incl.
    junk) for each. ``junk_suspects`` (pass 1's, post-enforcement) feeds the
    continuity seam test's flagged-break signal; omit for a caller that has
    none (continuity then just never reports a flagged break). ``synced_file_ids``
    (audio_sync.plan.md SS7.6): file_ids in a synced multicam group, whose
    continuity seam test skips the video shot-check -- see
    ``_has_scene_or_transition``'s ``synced`` param; None/empty is identical
    to today's behavior. ``sync_info_by_file`` (av_coupling_authoritative.
    plan.md): ``sync.store.sync_groups_for_files``'s own shape (``{file_id:
    {"authoritative_audio_file_id", "members": {file_id: {"offset_ms",
    ...}}}}``) -- fed straight through to ``sync.av_couple.authoritative_for``
    /``refine_offset`` to bake each cut's coupled ``(audio_file_id,
    audio_offset_ms)``. None/empty -> every cut couples to its own file at
    offset 0 (today's behavior). ``v4_meta_by_ref`` (cuts_v4_segmentation.
    plan.md): {source_ref: {"src_in_ms", "src_out_ms", "salience", "density"}}
    for a V4 ingest's video cuts -- when present for a cut, its span and
    salience come straight from the segmenter (v4_segment.VideoCut) instead
    of atom membership / post._salience, and its pace envelope's min_ms is
    density-scaled instead of anchor-derived. None/empty -> every video cut
    resolves exactly as today (V3). Raises ``ValueError`` (stage ``post``,
    per the plan's "no fallback" rule) on any invariant violation -- the
    caller marks the ingest run ``failed`` for re-run."""
    junk_by_file: Dict[str, List[JunkSuspect]] = {}
    for js in (junk_suspects or []):
        junk_by_file.setdefault(js.file_id, []).append(js)

    audio_by_file = audio_by_file or {}
    sync_info_by_file = sync_info_by_file or {}
    # Clip-relative normalisers for the quality scores: each signal is ranked
    # against its OWN clip's spread, so there are no absolute dB/pixel constants.
    rms_lohi = {fid: _series_lohi((a or {}).get("rms_db") or []) for fid, a in audio_by_file.items()}
    blur_lohi = {fid: _series_lohi((m or {}).get("blur") or []) for fid, m in motion_by_file.items()}
    ae_lohi = {fid: _series_lohi((m or {}).get("action_energy") or []) for fid, m in motion_by_file.items()}

    resolved: List[Tuple[Pass2Cut, int, int, List[int]]] = []
    for cut in pass2_output.cuts:
        lattice = lattices.get(cut.file_id)
        if lattice is None:
            raise ValueError(f"assemble_cut_records: unknown file_id {cut.file_id!r} ({cut.source_ref})")

        if cut.kind == "speech":
            silences = silences_by_file.get(cut.file_id, [])
            s, e = resolve_speech_span_ms(lattice.words, lattice.atoms, cut.word_span, silences)
        elif v4_meta_by_ref and cut.source_ref in v4_meta_by_ref:
            # cuts_v4_segmentation.plan.md: the segmenter's own tight span --
            # NOT the bounding box of the (coarser, informational-only) atoms
            # it happens to overlap. See v4_segment.segment_video's docstring
            # for why atom_ids can't drive this resolution for a V4 cut.
            meta = v4_meta_by_ref[cut.source_ref]
            s, e = int(meta["src_in_ms"]), int(meta["src_out_ms"])
        else:
            atoms_by_id = {a.atom_id: a for a in lattice.atoms}
            members = [atoms_by_id[i] for i in (cut.atom_ids or []) if i in atoms_by_id]
            if not members:
                raise ValueError(f"assemble_cut_records: no resolvable atoms for {cut.source_ref} "
                                 f"in {cut.file_id}")
            s = min(a.start_ms for a in members)
            e = max(a.end_ms for a in members)

        motion = motion_by_file.get(cut.file_id, {})
        anchors = _anchors_in(motion, s, e)
        resolved.append((cut, s, e, anchors))

    by_file: Dict[str, List[int]] = {}
    for idx, (cut, *_rest) in enumerate(resolved):
        by_file.setdefault(cut.file_id, []).append(idx)

    # A file with zero assigned cuts is now LEGAL (boundaries-v2): a clip that's
    # all dead air / all junk contributes nothing, and that's a valid outcome,
    # not a failure. Still worth a warning -- it's ALSO what a silently truncated
    # pass-2 response looks like (see llm.client._truncated), so a surprise empty
    # file is something to eyeball, just not something to abort the run over.
    missing = set(lattices.keys()) - set(by_file.keys())
    if missing:
        logger.warning("no cuts assigned for file(s) %s -- all-junk clip, or a "
                        "pass-2 omission worth checking", sorted(missing))

    synced_ids = synced_file_ids or set()
    next_start: Dict[int, int] = {}
    continuity_by_idx: Dict[int, Dict[str, Any]] = {}
    for file_id, idxs in by_file.items():
        idxs.sort(key=lambda i: resolved[i][1])
        duration_ms = lattices[file_id].duration_ms
        spans = [(resolved[i][1], resolved[i][2]) for i in idxs]
        _validate_no_overlap(file_id, spans, duration_ms)
        for pos, i in enumerate(idxs):
            next_start[i] = resolved[idxs[pos + 1]][1] if pos + 1 < len(idxs) else duration_ms
        junk_spans = _junk_suspect_spans(junk_by_file.get(file_id, []), lattices[file_id])
        continuity_by_idx.update(_clip_continuity(
            file_id, idxs, resolved, lattices[file_id], junk_spans,
            motion_by_file.get(file_id), v4_meta_by_ref, synced=file_id in synced_ids,
        ))

    out: List[CutRecord] = []
    for idx, (cut, s, e, anchors) in enumerate(resolved):
        motion = motion_by_file.get(cut.file_id, {})
        blur = motion.get("blur") or []
        hop_ms = int(motion.get("hop_ms") or 0)
        action_energy = motion.get("action_energy") or []
        audio = audio_by_file.get(cut.file_id) or {}
        v4_meta = (v4_meta_by_ref or {}).get(cut.source_ref) if cut.kind == "video" else None
        hero_ts = pick_hero_ts_ms(anchors, blur, hop_ms, s, e)
        pace = compute_pace_envelope(
            kind=cut.kind, s=s, e=e, readability_ms=cut.readability_ms, anchors=anchors,
            action_energy=action_energy, hop_ms=hop_ms,
            next_cut_start_ms=next_start[idx],
            max_tasteful_speed=cut.taste_fences.max_tasteful_speed,
            min_tasteful_speed=cut.taste_fences.min_tasteful_speed,
            natural_sound=cut.natural_sound,
            density=(v4_meta.get("density") if v4_meta is not None else None),
        )
        speech_quality: Optional[float] = None
        if cut.kind == "speech":
            pace.remove_spans = compute_speech_remove_spans(
                lattices[cut.file_id].words, cut.word_span, s, e)
            remove_budget = sum(b - a for a, b in pace.remove_spans)
            rms_lo, rms_hi = rms_lohi.get(cut.file_id, (None, None))
            speech_quality = compute_speech_quality(
                audio.get("rms_db") or [], int(audio.get("hop_ms") or 0),
                s, e, remove_budget, rms_lo, rms_hi)

        framing_dict = cut.framing.model_dump()
        look_dict = cut.look.model_dump()
        blur_lo, blur_hi = blur_lohi.get(cut.file_id, (None, None))
        visual_score = compute_visual_score(
            cut.on_camera, framing_dict, look_dict, blur, hop_ms, s, e, blur_lo, blur_hi)
        total_quality = compute_total_quality(cut.kind, speech_quality, visual_score)

        if v4_meta is not None:
            # cuts_v4_segmentation.plan.md section 4: the segmenter already
            # computed the novelty curve and the anchor kind -- emit its
            # salience directly rather than recomputing an absolute-level
            # peak in post._salience. `shape` (pass 2's coarse semantic
            # prior, arbitrated by construction -- see cutrecord_map._video_
            # rung's branch order: `kind` is always code-owned and decides
            # point/span/none; `shape` only ever picks the ladder's
            # before/after asymmetry, defaulting safely to symmetric
            # otherwise) rides alongside it.
            salience = dict(v4_meta.get("salience") or {})
            salience["shape"] = cut.shape
        else:
            ae_lo, ae_hi = ae_lohi.get(cut.file_id, (None, None))
            rms_lo, rms_hi = rms_lohi.get(cut.file_id, (None, None))
            salience = _salience(
                action_energy, hop_ms, s, e, ae_lo, ae_hi,
                audio.get("rms_db") or [], int(audio.get("hop_ms") or 0), rms_lo, rms_hi,
                anchors, audio.get("onsets_ms") or [], hero_ts,
            )

        # av_coupling_authoritative.plan.md: bake this cut's coupled audio
        # source NOW (assembly time), never re-derived lazily at render time.
        # Same-source (no group, or this file already IS the authoritative
        # source) -> identity coupling, zero cost. Cross-source -> the
        # group's global delta, LOCALLY REFINED against this cut's own
        # audio window so a loose global offset / long-take clock drift
        # can't show up as visible lip-sync error.
        audio_file_id, audio_global_delta = av_couple.authoritative_for(cut.file_id, sync_info_by_file)
        if audio_file_id == cut.file_id:
            audio_offset_ms, audio_align_confidence = 0, None
        else:
            auth_audio = audio_by_file.get(audio_file_id) or {}
            audio_offset_ms, audio_align_confidence = av_couple.refine_offset(
                audio.get("rms_db") or [], auth_audio.get("rms_db") or [],
                int(audio.get("hop_ms") or 0), s, e, audio_global_delta,
            )

        out.append(CutRecord(
            file_id=cut.file_id, src_in_ms=s, src_out_ms=e, kind=cut.kind,
            word_span=cut.word_span, atom_ids=cut.atom_ids, label=cut.label, summary=cut.summary,
            on_camera=cut.on_camera,
            junk=cut.junk, junk_reason=cut.junk_reason,
            framing=framing_dict, look=look_dict,
            caption_zones=list(cut.caption_zones), hero_ts_ms=hero_ts, pace=pace,
            take_group_id=cut.take_group_id, take_role=cut.take_role,
            speech_quality=speech_quality, total_quality=total_quality,
            characteristics=list(cut.people or []),
            # Channel is a SEMANTIC category the model owns: speech is always
            # "said" (code owns that fact); a video cut is "done" (an action is
            # performed/demonstrated) or "shown" (b-roll/display). Missing/unknown
            # on a video cut resolves to the conservative "shown".
            channel=("said" if cut.kind == "speech"
                     else (cut.channel if cut.channel in ("done", "shown") else "shown")),
            continuity=continuity_by_idx.get(idx, {}),
            camera=classify_camera_move(motion, s, e),
            sync_group_id=(sync_group_by_file or {}).get(cut.file_id),
            screen_text=cut.screen_text, salience=salience,
            voice_ids=cut.voice_ids, speaker_person=cut.speaker_person,
            visible_persons=cut.visible_persons,
            audio_file_id=audio_file_id, audio_offset_ms=audio_offset_ms,
            audio_align_confidence=audio_align_confidence,
        ))
    _enforce_take_winner(out)
    return out


def _enforce_take_winner(records: List[CutRecord]) -> None:
    """Deterministically crown the winner of every take group (in place),
    replacing pass 2's own winner call entirely -- the model owns the semantic
    grouping (which cuts are takes of one beat, which are outlook angles), code
    owns the numeric pick.

    A "winner" only ever means the best of a SAME-SETTING take cluster (retries
    of the same shot). Within that cluster the highest total_quality wins (ties
    to the longest, most complete take); the rest become plain 'take'. OUTLOOKS
    -- different-angle members of the group -- are never a winner: they are peer
    angles the arranger chooses between per beat using total_quality, not a
    crowned keeper. So if a group's only same-setting members number just one
    (no genuine retry) it is treated as another angle and left winner-less."""
    from collections import defaultdict
    groups: Dict[str, List[CutRecord]] = defaultdict(list)
    for r in records:
        if r.take_group_id:
            groups[r.take_group_id].append(r)
    for gid, members in groups.items():
        # winner/take == same-setting cluster; outlook == alternate angle.
        cluster = [m for m in members if m.take_role in ("winner", "take")]
        has_outlooks = any(m.take_role == "outlook" for m in members)
        if len(cluster) >= 2:
            best = max(cluster, key=lambda m: (m.total_quality, m.src_out_ms - m.src_in_ms))
            for m in cluster:
                m.take_role = "winner" if m is best else "take"
        elif len(cluster) == 1:
            # A lone same-setting member is not a take contest. If real outlook
            # angles sit alongside it, it IS just another angle -> demote to
            # outlook so no winner is named among peer angles. Otherwise leave it.
            cluster[0].take_role = "outlook" if has_outlooks else cluster[0].take_role
        # outlooks are never touched -> never a winner.
