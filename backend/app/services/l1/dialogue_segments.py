"""
L1 derived signal: dialogue segments (the "Dialogues" lens).

Turns the word-level transcript (+ per-word speaker from diarization) into clean,
drop-to-timeline speech selects at TWO granularities:

  * sentence -- a single utterance/sentence (tight, social-media style).
  * topic    -- a complete thought/answer; merges a speaker's consecutive
                sentences, bridging the other speaker's short backchannel
                ("mhm", "yeah") so an answer isn't chopped by the interviewer.

The craft is in WHERE we cut. We never cut on Whisper word timestamps (their
END times truncate trailing consonants); instead we compute a FINE energy
envelope from the WAV and snap every in/out to the quietest point (silence
trough) in a small search window, with handles + a short fade so a clip drops
in click-free. Speaker change is always a hard boundary -- two speakers become
two adjacent clips, never one merged blob. Overlapping cross-talk is flagged,
not guessed.

Pure-Python/numpy + one librosa load. CPU, best-effort: any failure returns an
empty result so the caller can treat the lens as optional.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# --- Phrase / sentence / topic thresholds (ms) ---------------------------
PHRASE_GAP_MS = 250        # gap that breaks a phrase (a breath)
G_SENTENCE_MS = 350        # max gap bridged inside one sentence
SENT_L_MAX_MS = 12_000     # cap a sentence clip length
SENT_L_MIN_MS = 1_200      # below this, try to merge a fragment forward
TOPIC_GAP_MS = 1_200       # gap that starts a new topic
BACKCHANNEL_MAX_MS = 1_000 # an interjection this short can be bridged

# --- Cut-point snapping (ms) ---------------------------------------------
PRE_MAX_MS = 150           # how far BEFORE the first word we look for silence
IN_FWD_MS = 60             # ...and how far into the word
OUT_BACK_MS = 60           # how far before the last word's end we look
POST_MAX_MS = 200          # ...and how far after
DEFAULT_HANDLE_MS = 100    # fallback handle when there's no clean trough
FADE_MS = 15               # audio de-click fade baked onto every clip
SILENCE_DROP_DB = 12.0     # a trough must dip this far below speech to be "clean"

# --- Lexicons ------------------------------------------------------------
SENTENCE_FINAL = (".", "?", "!", "…")
BACKCHANNEL_WORDS = {
    "mhm", "mm", "mmhm", "uhhuh", "yeah", "yep", "yes", "right", "okay", "ok",
    "sure", "exactly", "totally", "gotcha", "wow", "nice", "cool", "hmm", "huh",
}
DISCOURSE_MARKERS = {
    "so", "okay", "ok", "anyway", "anyways", "now", "next", "alright",
    "well", "basically", "another", "firstly", "secondly", "finally", "lastly",
}

# --- Off-camera / production-cue suppression (#1 lexicon, #2 loudness) ----
# Words a crew member shouts that are NOT part of the on-camera dialogue. We
# only suppress these when they're SHORT and ISOLATED (and, for the ambiguous
# ones, near a clip edge) so we never eat a real line that happens to contain
# "go" or "start". STRONG cues are almost never real dialogue; WEAK cues are
# common words, so they need the edge gate too.
STRONG_CUE_WORDS = {"action", "cut", "rolling", "slate", "marker", "speed"}
WEAK_CUE_WORDS = {"go", "start", "stop", "ready", "set", "begin", "reset", "again", "take"}
CUE_MAX_MS = 1_200          # a cue is a brief shout, not a sentence
CUE_MAX_WORDS = 3
CUE_EDGE_MS = 2_000         # "near a clip edge" window for the ambiguous cues
CUE_ISOLATION_MS = 600      # silence to a neighbour that marks a unit as isolated
# A unit whose speech sits this far below the clip's speech reference is likely
# off-mic (someone behind the camera) rather than the on-camera subject.
OFFSCREEN_LEVEL_DROP_DB = 12.0


@dataclass
class _Unit:
    """A speaker-pure span before snapping (phrase / sentence / topic)."""
    speaker: Optional[str]
    raw_in_ms: int
    raw_out_ms: int
    text: str
    is_backchannel: bool = False
    children: List[int] = field(default_factory=list)  # indices into the level below
    flags: List[str] = field(default_factory=list)  # e.g. "production_cue", "offscreen"


# ---------------------------------------------------------------------------
# Fine energy envelope (for silence-snapping)
# ---------------------------------------------------------------------------

class Envelope:
    """Fine RMS-in-dB envelope at `hop_ms` resolution, with a trough finder.

    `from_wav` builds it from the 16k mono WAV; `silent` is a degenerate
    fallback (no audio) that snaps to the window midpoint so the pipeline
    still produces usable cuts.
    """

    def __init__(self, rms_db, hop_ms: int, speech_ref: float):
        self.rms_db = rms_db
        self.hop_ms = max(1, int(hop_ms))
        self.speech_ref = speech_ref
        self.n = len(rms_db)

    @classmethod
    def from_wav(cls, wav_path: str) -> "Envelope":
        import librosa
        import numpy as np

        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if y.size == 0:
            return cls.silent()
        hop = 160          # 10ms at 16kHz
        frame = 320        # 20ms window
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        if rms.size == 0:
            return cls.silent()
        rms_db = 20.0 * np.log10(rms + 1e-6)
        speech_ref = float(np.percentile(rms_db, 90))
        return cls(rms_db, hop_ms=int(1000 * hop / sr), speech_ref=speech_ref)

    @classmethod
    def silent(cls) -> "Envelope":
        return cls([], hop_ms=10, speech_ref=0.0)

    def _idx(self, ms: int) -> int:
        return max(0, min(self.n - 1, int(round(ms / self.hop_ms))))

    def trough(self, lo_ms: int, hi_ms: int) -> Tuple[int, float, bool]:
        """Return (time_ms, db_at_trough, is_clean) for the quietest point in
        [lo_ms, hi_ms]. `is_clean` = the trough dips clearly below speech."""
        if self.n == 0 or hi_ms <= lo_ms:
            return (lo_ms + hi_ms) // 2, 0.0, False
        a, b = self._idx(lo_ms), self._idx(hi_ms)
        if b <= a:
            return lo_ms, float(self.rms_db[a]), False
        best_i = a
        best_v = self.rms_db[a]
        for i in range(a, b + 1):
            if self.rms_db[i] < best_v:
                best_v = self.rms_db[i]
                best_i = i
        clean = best_v <= self.speech_ref - SILENCE_DROP_DB
        return best_i * self.hop_ms, float(best_v), clean

    def level_db(self, lo_ms: int, hi_ms: int) -> Optional[float]:
        """Median speech level (dB) over [lo_ms, hi_ms]; None if no audio.
        Used to spot off-mic (quiet) speech relative to the clip's speech_ref."""
        if self.n == 0 or hi_ms <= lo_ms:
            return None
        import numpy as np
        a, b = self._idx(lo_ms), self._idx(hi_ms)
        if b <= a:
            return None
        return float(np.median(self.rms_db[a:b + 1]))


# ---------------------------------------------------------------------------
# Word helpers
# ---------------------------------------------------------------------------

def _norm(token: str) -> str:
    return "".join(ch for ch in (token or "").lower() if ch.isalpha())


def _join_text(words: Sequence[dict]) -> str:
    return " ".join((w.get("text") or "").strip() for w in words if (w.get("text") or "").strip())


def _content_words(words: Sequence[dict]) -> List[dict]:
    """Non-filler words; falls back to all words if everything is a filler
    (e.g. a bare 'mm' backchannel) so the span is never empty."""
    content = [w for w in words if not w.get("is_filler")]
    return content or list(words)


def _span(words: Sequence[dict]) -> Tuple[int, int]:
    s = min(int(w.get("start_ms", 0)) for w in words)
    e = max(int(w.get("end_ms", 0)) for w in words)
    return s, max(e, s)


# ---------------------------------------------------------------------------
# Phrases (atoms)
# ---------------------------------------------------------------------------

def build_phrases(words: Sequence[dict]) -> List[List[dict]]:
    """Group consecutive words into phrases. A new phrase starts on a speaker
    change or a gap > PHRASE_GAP_MS. Each phrase is a list of word dicts."""
    phrases: List[List[dict]] = []
    cur: List[dict] = []
    prev_end: Optional[int] = None
    prev_spk: Optional[str] = None
    for w in words:
        s = int(w.get("start_ms", 0))
        spk = w.get("speaker")
        brk = False
        if cur:
            if spk != prev_spk:
                brk = True
            elif prev_end is not None and s - prev_end > PHRASE_GAP_MS:
                brk = True
        if brk:
            phrases.append(cur)
            cur = []
        cur.append(w)
        prev_end = int(w.get("end_ms", s))
        prev_spk = spk
    if cur:
        phrases.append(cur)
    return phrases


def _unit_from_words(words: Sequence[dict]) -> _Unit:
    content = _content_words(words)
    raw_in, raw_out = _span(content)
    text = _join_text(content)
    dur = raw_out - raw_in
    norm_all = [_norm(w.get("text", "")) for w in content]
    is_bc = (
        dur <= BACKCHANNEL_MAX_MS
        and 0 < len(norm_all) <= 3
        and all(t in BACKCHANNEL_WORDS or t == "" for t in norm_all)
        and any(norm_all)
    )
    return _Unit(
        speaker=words[0].get("speaker") if words else None,
        raw_in_ms=raw_in, raw_out_ms=raw_out, text=text, is_backchannel=is_bc,
    )


# ---------------------------------------------------------------------------
# Sentence units
# ---------------------------------------------------------------------------

def merge_sentences(words: Sequence[dict]) -> List[_Unit]:
    """Group words into sentence units. A new sentence starts on a speaker
    change, a gap > G_SENTENCE_MS, after sentence-final punctuation, or at the
    length cap. Word-level (not phrase-level) so punctuation inside one breath
    still splits two sentences."""
    units: List[_Unit] = []
    cur: List[dict] = []

    def flush():
        if cur:
            units.append(_unit_from_words(cur))

    prev_end: Optional[int] = None
    prev_spk: Optional[str] = None
    for w in words:
        s = int(w.get("start_ms", 0))
        spk = w.get("speaker")
        if cur:
            gap = s - (prev_end or s)
            cur_s, _ = _span(cur)
            too_long = (int(w.get("end_ms", s)) - cur_s) > SENT_L_MAX_MS
            last_txt = (cur[-1].get("text") or "").strip()
            sentence_end = last_txt.endswith(SENTENCE_FINAL)
            if spk != prev_spk or gap > G_SENTENCE_MS or sentence_end or too_long:
                flush()
                cur = []
        cur.append(w)
        prev_end = int(w.get("end_ms", s))
        prev_spk = spk
    flush()

    return _merge_short_forward(units)


def _merge_short_forward(units: List[_Unit]) -> List[_Unit]:
    """Fold a sub-SENT_L_MIN fragment into the next unit when it's the same
    speaker and close, so we don't emit 300ms shards. Backchannels are left
    alone (they're meaningful as-is)."""
    out: List[_Unit] = []
    i = 0
    while i < len(units):
        u = units[i]
        dur = u.raw_out_ms - u.raw_in_ms
        complete = u.text.strip().endswith(SENTENCE_FINAL)
        if (
            dur < SENT_L_MIN_MS and not u.is_backchannel and not complete
            and i + 1 < len(units)
            and units[i + 1].speaker == u.speaker
            and units[i + 1].raw_in_ms - u.raw_out_ms <= G_SENTENCE_MS
        ):
            nxt = units[i + 1]
            merged = _Unit(
                speaker=u.speaker,
                raw_in_ms=u.raw_in_ms,
                raw_out_ms=nxt.raw_out_ms,
                text=(u.text + " " + nxt.text).strip(),
            )
            out.append(merged)
            i += 2
            continue
        out.append(u)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Off-camera / production-cue marking
# ---------------------------------------------------------------------------

def _is_production_cue(
    u: _Unit, prev_out: Optional[int], next_in: Optional[int],
    clip_start: int, clip_end: int,
) -> bool:
    """A short, isolated crew cue ('action', 'go', 'cut') rather than dialogue.
    Strong cues fire when isolated anywhere; ambiguous (weak) cues additionally
    require sitting near a clip edge, so a mid-sentence 'go' is never touched."""
    toks = [t for t in (_norm(w) for w in u.text.split()) if t]
    if not toks or len(toks) > CUE_MAX_WORDS:
        return False
    if (u.raw_out_ms - u.raw_in_ms) > CUE_MAX_MS:
        return False
    gap_before = (u.raw_in_ms - prev_out) if prev_out is not None else 10 ** 9
    gap_after = (next_in - u.raw_out_ms) if next_in is not None else 10 ** 9
    if max(gap_before, gap_after) < CUE_ISOLATION_MS:
        return False  # nestled between other speech -> almost certainly real
    near_edge = (
        (u.raw_in_ms - clip_start) <= CUE_EDGE_MS
        or (clip_end - u.raw_out_ms) <= CUE_EDGE_MS
    )
    if all(t in STRONG_CUE_WORDS for t in toks):
        return True
    if near_edge and all(t in (STRONG_CUE_WORDS | WEAK_CUE_WORDS) for t in toks):
        return True
    return False


def _mark_offscreen_units(
    units: List[_Unit], env: Envelope, clip_start: int, clip_end: int
) -> None:
    """In-place: tag units that look like off-camera speech. `production_cue`
    (lexicon) and `offscreen` (well below the clip's speech level) -- both kept
    out of topics and hidden by default in the UI, but never deleted."""
    for i, u in enumerate(units):
        prev_out = units[i - 1].raw_out_ms if i > 0 else None
        next_in = units[i + 1].raw_in_ms if i + 1 < len(units) else None
        if _is_production_cue(u, prev_out, next_in, clip_start, clip_end):
            u.flags.append("production_cue")
            continue
        if env.speech_ref == 0.0:
            continue  # silent fallback -> no usable level reference
        gap_before = (u.raw_in_ms - prev_out) if prev_out is not None else 10 ** 9
        gap_after = (next_in - u.raw_out_ms) if next_in is not None else 10 ** 9
        isolated = max(gap_before, gap_after) >= CUE_ISOLATION_MS
        near_edge = (
            (u.raw_in_ms - clip_start) <= CUE_EDGE_MS
            or (clip_end - u.raw_out_ms) <= CUE_EDGE_MS
        )
        if not (isolated or near_edge):
            continue  # only judge loudness on isolated/edge bits, not mid-convo
        lvl = env.level_db(u.raw_in_ms, u.raw_out_ms)
        if lvl is not None and lvl <= env.speech_ref - OFFSCREEN_LEVEL_DROP_DB:
            u.flags.append("offscreen")


# ---------------------------------------------------------------------------
# Topic units
# ---------------------------------------------------------------------------

def build_topics(sentences: List[_Unit]) -> List[_Unit]:
    """Merge a speaker's consecutive sentences into one topic, bridging the
    other speaker's short backchannel. Starts a new topic on a real speaker
    change, a long pause, or a discourse marker after a pause. Backchannels do
    not appear as their own topics."""
    topics: List[_Unit] = []
    cur: List[int] = []          # indices into `sentences` for the main speaker
    cur_spk: Optional[str] = None
    last_end: Optional[int] = None

    def flush():
        nonlocal cur
        if cur:
            first = sentences[cur[0]]
            last = sentences[cur[-1]]
            topics.append(_Unit(
                speaker=cur_spk,
                raw_in_ms=first.raw_in_ms,
                raw_out_ms=last.raw_out_ms,
                text=" ".join(sentences[i].text for i in cur).strip(),
                children=list(cur),
            ))
        cur = []

    for idx, s in enumerate(sentences):
        # Off-camera speech / crew cues never seed or join a topic.
        if "production_cue" in s.flags or "offscreen" in s.flags:
            continue
        # A short interjection by the OTHER speaker doesn't end the topic.
        if cur and s.is_backchannel and s.speaker != cur_spk:
            continue
        if not cur:
            cur = [idx]
            cur_spk = s.speaker
            last_end = s.raw_out_ms
            continue

        gap = s.raw_in_ms - (last_end or s.raw_in_ms)
        first_word = _norm(s.text.split(" ", 1)[0]) if s.text else ""
        marker_break = gap > G_SENTENCE_MS and first_word in DISCOURSE_MARKERS
        if s.speaker != cur_spk or gap > TOPIC_GAP_MS or marker_break:
            flush()
            cur = [idx]
            cur_spk = s.speaker
            last_end = s.raw_out_ms
        else:
            cur.append(idx)
            last_end = s.raw_out_ms
    flush()
    return topics


# ---------------------------------------------------------------------------
# Snapping units -> segments
# ---------------------------------------------------------------------------

def _snap_in(env: Envelope, raw_in: int, prev_out: Optional[int]) -> Tuple[int, bool]:
    lo = raw_in - PRE_MAX_MS
    if prev_out is not None:
        lo = max(lo, prev_out + 1)
    lo = max(0, lo)
    hi = raw_in + IN_FWD_MS
    t, _v, clean = env.trough(lo, hi)
    if clean:
        return t, False
    cut = max(0, raw_in - DEFAULT_HANDLE_MS)
    if prev_out is not None:
        cut = max(cut, prev_out + 1)
    return cut, True


def _snap_out(env: Envelope, raw_out: int, next_in: Optional[int]) -> Tuple[int, bool]:
    lo = raw_out - OUT_BACK_MS
    hi = raw_out + POST_MAX_MS
    if next_in is not None:
        hi = min(hi, next_in - 1)
    if hi <= lo:
        hi = raw_out
    t, _v, clean = env.trough(lo, hi)
    if clean:
        return t, False
    cut = raw_out + DEFAULT_HANDLE_MS
    if next_in is not None:
        cut = min(cut, next_in - 1)
    return cut, True


def _snap_units(units: List[_Unit], env: Envelope, level: str) -> List[dict]:
    segs: List[dict] = []
    for i, u in enumerate(units):
        prev_out = units[i - 1].raw_out_ms if i > 0 else None
        next_in = units[i + 1].raw_in_ms if i + 1 < len(units) else None
        src_in, noisy_in = _snap_in(env, u.raw_in_ms, prev_out)
        src_out, noisy_out = _snap_out(env, u.raw_out_ms, next_in)
        if src_out <= src_in:
            src_out = max(u.raw_out_ms, src_in + 1)
        flags: List[str] = list(u.flags)
        if noisy_in or noisy_out:
            flags.append("noisy")
        if u.is_backchannel:
            flags.append("backchannel")
        segs.append({
            "seg_id": f"{level}-{i}",
            "level": level,
            "order": i,
            "speaker": u.speaker,
            "text": u.text,
            "src_in_ms": int(src_in),
            "src_out_ms": int(src_out),
            "raw_in_ms": int(u.raw_in_ms),
            "raw_out_ms": int(u.raw_out_ms),
            "fade_in_ms": FADE_MS,
            "fade_out_ms": FADE_MS,
            "topic_id": None,
            "child_seg_ids": [],
            "flags": flags,
            "confidence": 1.0,
        })
    _flag_overlaps(segs)
    return segs


def _flag_overlaps(segs: List[dict]) -> None:
    """Adjacent different-speaker selects whose raw spans overlap = cross-talk."""
    for a, b in zip(segs, segs[1:]):
        if a["speaker"] != b["speaker"] and a["raw_out_ms"] > b["raw_in_ms"]:
            for s in (a, b):
                if "overlap" not in s["flags"]:
                    s["flags"].append("overlap")


def _link_topics(sentence_segs: List[dict], topic_segs: List[dict], topic_units: List[_Unit]) -> None:
    """Stamp topic_id onto both levels and list each topic's child sentences."""
    for t_idx, (tseg, tunit) in enumerate(zip(topic_segs, topic_units)):
        tseg["topic_id"] = t_idx
        children = [
            sentence_segs[ci]["seg_id"]
            for ci in tunit.children
            if 0 <= ci < len(sentence_segs)
        ]
        tseg["child_seg_ids"] = children
        for ci in tunit.children:
            if 0 <= ci < len(sentence_segs):
                sentence_segs[ci]["topic_id"] = t_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dialogue_segments(words: Sequence[dict], wav_path: Optional[str]) -> Dict[str, List[dict]]:
    """Return {"sentence": [...], "topic": [...]} for a file's words.

    `words` is the flat, chronological word list (start_ms/end_ms/text/
    is_filler/speaker). `wav_path` enables silence-snapped cuts; if None or
    unreadable we fall back to handle-based cuts. Never raises."""
    out: Dict[str, List[dict]] = {"sentence": [], "topic": []}
    flat = [w for w in (words or []) if (w.get("text") or "").strip()]
    if not flat:
        return out

    try:
        env = Envelope.from_wav(wav_path) if wav_path else Envelope.silent()
    except Exception:
        logger.exception("Dialogue envelope failed for %s; using silent fallback.", wav_path)
        env = Envelope.silent()

    clip_start = min(int(w.get("start_ms", 0)) for w in flat)
    clip_end = max(int(w.get("end_ms", 0)) for w in flat)
    clip_end = max(clip_end, clip_start)

    sentence_units = merge_sentences(flat)
    _mark_offscreen_units(sentence_units, env, clip_start, clip_end)
    topic_units = build_topics(sentence_units)
    _mark_offscreen_units(topic_units, env, clip_start, clip_end)

    sentence_segs = _snap_units(sentence_units, env, "sentence")
    topic_segs = _snap_units(topic_units, env, "topic")
    _link_topics(sentence_segs, topic_segs, topic_units)

    out["sentence"] = sentence_segs
    out["topic"] = topic_segs
    return out
