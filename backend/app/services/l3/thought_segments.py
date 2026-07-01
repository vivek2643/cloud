"""
L3 thought segmentation: the speech primitive the energy bands cut from.

WHY this exists. The energy bands for speech used to be built off the L1
``dialogue_segments`` SENTENCE/TOPIC units -- boundaries drawn from ASR
punctuation + pause gaps. Those edges are ragged (mid-sentence starts, a
trailing clause orphaned, a real thought chopped where the speaker drew breath),
so *no* downstream brain could assemble coherent speech from them. The fix is to
let a strong model do what an editor does first: read the whole transcript and
break it into THOUGHTS.

WHAT a thought is (deliberately GENERIC -- no interview/Q&A semantics baked in):
one speaker's single, self-contained idea, with a zoom hierarchy so the energy
dial has clean material at every level:

    punchline   -- the tightest clause that lands the point        (Sharp)
    core        -- the one sentence that carries it                (Tight)
    thought     -- the complete idea, start to end                 (Balanced)
    + setup     -- the same speaker's own run-up into it           (Calm)
    (whole turn -- every consecutive thought by the speaker)       (Broad)

The "turn" level is DERIVED downstream (consecutive same-speaker thoughts), so it
is not part of the stored shape. There is no ``kind`` (question/answer) and no
cross-thought link: question/answer composition emerges because the brain reads
the transcript in order, with speaker labels -- exactly how a human picks.

WHERE it runs. After L2, in the cut layer, alongside the hero-cuts precompute --
so it sees the L1 transcript + diarized speaker AND (later) L2 signals, and the
cut it feeds carries file context like every other modality. Computed ONCE per
file and cached (``speech_thoughts``); a stale/missing entry recomputes lazily.

Fails OPEN: any LLM/parse error (or an over-long clip) degrades to thoughts
derived deterministically from the existing L1 ``dialogue_segments``, so the
speech path never loses its source.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.l3 import score_span as ss
from app.services.llm import LLMClient, get_llm, user_message

logger = logging.getLogger(__name__)

# Bump when the thought shape or the prompt contract changes, so cached rows
# recompute. Combined with the clip's word count into the cache signature.
# v2: prompt keeps questions as thoughts + allows degenerate depth (punch==core
# ==thought for one-liners); the speech cut now owns its full zoom ladder.
# v3: SILENCE CEILING -- the word stream now carries pause markers so the model
# won't span a long dead gap, and every level is deterministically clamped to
# the pause-bounded run containing its punch (a spoken thought can't hold a big
# hole of silence). Speech-only by construction -- a demo's silence lives in the
# video channels, not here.
THOUGHT_SCHEMA_VERSION = 3

# A thought needs at least this many real words to be worth keeping as a unit.
_MIN_THOUGHT_WORDS = 3

# The silence ceiling for a spoken thought. A dead gap longer than this between
# two words is not natural phrasing -- it is the boundary between two thoughts
# (the speaker stopped, a demo happened, the camera cut). No single said cut may
# straddle it: every level is clamped to the pause-bounded run around its punch.
_SILENCE_MAX_MS = 1500
# Gaps at/above this are surfaced to the model as an explicit ``pause`` marker in
# the word stream, so it segments across them instead of merging (the marker is
# informational -- it does not consume a word index).
_PAUSE_MARK_MS = 800


# --------------------------------------------------------------------------
# The thought shape
# --------------------------------------------------------------------------

@dataclass
class Span:
    """A word-index range + the millisecond window it resolves to.

    ``start_word``/``end_word`` are indices into the clip's flat, non-empty word
    list (inclusive); they're ``None`` for the L1 fallback, which works in ms
    only. ``raw_in_ms``/``raw_out_ms`` are the unsnapped bounds (the fused-seam
    snap is applied later, in the hero-cuts speech path)."""
    raw_in_ms: int
    raw_out_ms: int
    text: str
    start_word: Optional[int] = None
    end_word: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        return cls(
            raw_in_ms=int(d.get("raw_in_ms", 0)),
            raw_out_ms=int(d.get("raw_out_ms", 0)),
            text=str(d.get("text") or ""),
            start_word=d.get("start_word"),
            end_word=d.get("end_word"),
        )


@dataclass
class Thought:
    """One speaker's self-contained idea + its zoom hierarchy."""
    speaker: Optional[str]
    thought: Span                 # Balanced -- the complete idea
    core: Span                    # Tight -- the one carrying sentence
    punch: Span                   # Sharp -- the tightest landing clause
    setup: Optional[Span]         # Calm extra -- the speaker's own run-up
    strength: float               # 0..1 how strong/usable as a standalone

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "thought": self.thought.to_dict(),
            "core": self.core.to_dict(),
            "punch": self.punch.to_dict(),
            "setup": self.setup.to_dict() if self.setup else None,
            "strength": round(float(self.strength), 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Thought":
        setup = d.get("setup")
        return cls(
            speaker=d.get("speaker"),
            thought=Span.from_dict(d.get("thought") or {}),
            core=Span.from_dict(d.get("core") or d.get("thought") or {}),
            punch=Span.from_dict(d.get("punch") or d.get("core") or d.get("thought") or {}),
            setup=Span.from_dict(setup) if setup else None,
            strength=float(d.get("strength", 0.5)),
        )


# --------------------------------------------------------------------------
# Word helpers
# --------------------------------------------------------------------------

def _real_words(source: ss.SpanSource) -> List[dict]:
    """The clip's flat, chronological, non-empty words (speaker preserved)."""
    return [w for w in (source.words or []) if (w.get("text") or "").strip()]


def _word_ms(words: List[dict], i: int, j: int) -> Tuple[int, int]:
    """(start_ms of word i, end_ms of word j), clamped to the list."""
    n = len(words)
    i = max(0, min(i, n - 1))
    j = max(i, min(j, n - 1))
    return int(words[i].get("start_ms", 0)), int(words[j].get("end_ms", 0))


def _word_text(words: List[dict], i: int, j: int) -> str:
    n = len(words)
    i = max(0, min(i, n - 1))
    j = max(i, min(j, n - 1))
    return " ".join((w.get("text") or "").strip() for w in words[i:j + 1]).strip()


def _span_from_words(words: List[dict], i: int, j: int) -> Span:
    in_ms, out_ms = _word_ms(words, i, j)
    return Span(raw_in_ms=in_ms, raw_out_ms=out_ms,
               text=_word_text(words, i, j),
               start_word=max(0, min(i, len(words) - 1)),
               end_word=max(0, min(j, len(words) - 1)))


def _gap_after(words: List[dict], i: int) -> int:
    """Silence between word ``i`` and word ``i+1`` (0 at the list edges)."""
    if i < 0 or i + 1 >= len(words):
        return 0
    return int(words[i + 1].get("start_ms", 0)) - int(words[i].get("end_ms", 0))


def _run_within(words: List[dict], lo_bound: int, hi_bound: int, anchor: int) -> Tuple[int, int]:
    """Widen out from ``anchor`` (a word index) as far as possible WITHOUT
    crossing a gap longer than ``_SILENCE_MAX_MS``, staying inside
    [lo_bound, hi_bound]. This is the pause-bounded run of speech the anchor sits
    in -- everything past the first long silence on either side is a DIFFERENT
    thought. Used to strip dead air out of a thought's levels."""
    lo = hi = max(lo_bound, min(anchor, hi_bound))
    while lo > lo_bound and _gap_after(words, lo - 1) <= _SILENCE_MAX_MS:
        lo -= 1
    while hi < hi_bound and _gap_after(words, hi) <= _SILENCE_MAX_MS:
        hi += 1
    return lo, hi


# --------------------------------------------------------------------------
# LLM pass
# --------------------------------------------------------------------------

_SYSTEM = (
    "You segment a transcript into THOUGHTS for a video editor. A thought is one "
    "speaker's single, self-contained idea -- what you'd keep as one coherent "
    "spoken beat. You never merge two speakers, and you cover the whole clip."
)

_INSTR = (
    "Below is one clip's transcript. Every word is numbered `i:word`; a `[Sx]` "
    "marks where the speaker changes; a `<pause Ns>` marks N seconds of silence "
    "between words. NEVER let one thought span a long pause (a `<pause>` of ~1.5s "
    "or more) -- that silence is a boundary BETWEEN thoughts (the speaker stopped, "
    "a demo happened, the shot changed), so start a new thought after it. Split "
    "the transcript into thoughts IN ORDER.\n\n"
    "For each thought give word indices (inclusive) into the numbered words:\n"
    "  - thought: the complete idea proper, start to end (one speaker) -- NOT "
    "counting any throat-clearing run-up.\n"
    "  - core: the single sentence inside `thought` that carries the point.\n"
    "  - punch: the tightest clause inside `core` that lands the point.\n"
    "  - setup: the SAME speaker's run-up that leads INTO the idea, if any, else "
    "null. It sits JUST BEFORE `thought` -- its words come before thought's first "
    "index, never inside it.\n"
    "  - strength: 0..1, how well it stands on its own as a clip.\n\n"
    "Rules: a thought is one speaker only; punch ⊆ core ⊆ thought; setup (if any) "
    "is entirely before thought; cover every spoken word across consecutive "
    "thoughts (a word is either a thought's setup or part of a thought); don't "
    "label types -- just segment.\n"
    "A QUESTION is a real thought (an interviewer's prompt, an aside) -- segment "
    "it like any other, never skip it. When an idea is already a single short "
    "line, the levels COLLAPSE: punch == core == thought is correct (don't invent "
    "a tighter clause that isn't there).\n\n"
    "Return ONLY this JSON:\n"
    "{\"thoughts\":[{\"speaker\":\"Sx\",\"thought\":[i,j],\"core\":[i,j],"
    "\"punch\":[i,j],\"setup\":[i,j]|null,\"strength\":0.0}]}\n\n"
    "TRANSCRIPT:\n"
)


def _render_words(words: List[dict]) -> str:
    """Numbered, speaker-marked word stream for the model, with explicit pause
    markers so a long silence reads as a thought boundary (the model is otherwise
    blind to time and would merge words across a dead gap)."""
    out: List[str] = []
    prev_spk = object()
    for i, w in enumerate(words):
        spk = w.get("speaker")
        if spk != prev_spk:
            out.append(f"\n[{spk or 'S?'}]")
            prev_spk = spk
        out.append(f"{i}:{(w.get('text') or '').strip()}")
        gap = _gap_after(words, i)
        if gap >= _PAUSE_MARK_MS:
            out.append(f"<pause {gap / 1000:.1f}s>")
    return " ".join(out).strip()


def _parse(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _pair(v: Any) -> Optional[Tuple[int, int]]:
    """Coerce a [i, j] index pair; None when malformed."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        i, j = int(v[0]), int(v[1])
    except (TypeError, ValueError):
        return None
    return (i, j) if j >= i else (j, i)


def _coerce_thought(raw: Any, words: List[dict]) -> Optional[Thought]:
    """One LLM thought dict -> a validated Thought, or None if unusable.

    Clamps every range into the thought, enforces punch ⊆ core ⊆ thought and
    setup before core, and drops thoughts too short to be a unit."""
    if not isinstance(raw, dict):
        return None
    th = _pair(raw.get("thought"))
    if th is None:
        return None
    n = len(words)
    ti, tj = max(0, th[0]), min(n - 1, th[1])
    if tj - ti + 1 < _MIN_THOUGHT_WORDS:
        return None

    core = _pair(raw.get("core")) or (ti, tj)
    ci, cj = max(ti, core[0]), min(tj, core[1])
    if cj < ci:
        ci, cj = ti, tj

    punch = _pair(raw.get("punch")) or (ci, cj)
    pi, pj = max(ci, punch[0]), min(cj, punch[1])
    if pj < pi:
        pi, pj = ci, cj

    # SILENCE CEILING (deterministic guarantee, independent of the model): a
    # spoken thought must sit inside ONE pause-bounded run. Clamp every level to
    # the run of speech around the punch, dropping any words the model reached
    # across a long dead gap (they are a separate thought). A no-op when the
    # model already segmented on the pauses it was shown; a hard backstop when it
    # didn't. Video demos keep their silence -- this ceiling is speech-only.
    rlo, rhi = _run_within(words, ti, tj, pi)
    ti, tj = rlo, rhi
    ci, cj = max(ci, rlo), min(cj, rhi)
    pi, pj = max(pi, rlo), min(pj, rhi)
    if tj - ti + 1 < _MIN_THOUGHT_WORDS:
        return None

    # Setup is the run-up BEFORE the thought (so Calm = setup + thought is a
    # strictly wider level than Balanced = thought). Clamp it to end just before
    # the thought's first word; drop it if nothing fits (e.g. thought at idx 0).
    setup_span: Optional[Span] = None
    setup = _pair(raw.get("setup"))
    if setup is not None and ti > 0:
        si, sj = max(0, setup[0]), min(ti - 1, setup[1])
        if sj >= si:
            setup_span = _span_from_words(words, si, sj)

    speaker = raw.get("speaker") or (words[ti].get("speaker") if ti < n else None)
    try:
        strength = float(raw.get("strength", 0.5))
    except (TypeError, ValueError):
        strength = 0.5
    strength = max(0.0, min(1.0, strength))

    return Thought(
        speaker=speaker,
        thought=_span_from_words(words, ti, tj),
        core=_span_from_words(words, ci, cj),
        punch=_span_from_words(words, pi, pj),
        setup=setup_span,
        strength=strength,
    )


def segment_with_llm(words: List[dict], llm: LLMClient) -> List[Thought]:
    """Run the LLM thought pass over a clip's words. Raises on transport error
    (the caller catches and falls back); returns [] only if the model returns
    nothing usable."""
    settings = get_settings()
    prompt = _INSTR + _render_words(words)
    resp = llm.run(system=_SYSTEM, messages=[user_message(prompt)],
                   max_tokens=settings.thoughts_max_output_tokens)
    data = _parse(resp.text) or {}
    raw = data.get("thoughts")
    if not isinstance(raw, list):
        return []
    out: List[Thought] = []
    for item in raw:
        t = _coerce_thought(item, words)
        if t is not None:
            out.append(t)
    return out


# --------------------------------------------------------------------------
# Deterministic fallback (from L1 dialogue_segments)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _load_dialogue(file_id: str) -> Dict[str, List[dict]]:
    """The L1 sentence/topic segments for one file (empty when none)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from dialogue_segments where file_id = %s",
            (file_id,),
        ).fetchone()
    if not row or not row[0]:
        return {"sentence": [], "topic": []}
    seg = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return {"sentence": seg.get("sentence") or [], "topic": seg.get("topic") or []}


def _span_from_seg(seg: dict) -> Span:
    return Span(
        raw_in_ms=int(seg.get("raw_in_ms", seg.get("src_in_ms", 0))),
        raw_out_ms=int(seg.get("raw_out_ms", seg.get("src_out_ms", 0))),
        text=(seg.get("text") or "").strip(),
    )


def segment_fallback(file_id: str) -> List[Thought]:
    """Derive thoughts from the L1 dialogue hierarchy (no model).

    Each TOPIC becomes a thought (the complete idea); its LAST child sentence is
    the core+punch (the payoff usually lands last). Setup is left null -- the L1
    hierarchy has no reliable notion of a same-speaker run-up, so Calm collapses
    to the thought here. Ragged, but a safe floor so the speech path always has a
    source."""
    dlg = _load_dialogue(file_id)
    topics = dlg.get("topic") or []
    sentences = {s.get("seg_id"): s for s in (dlg.get("sentence") or [])}
    out: List[Thought] = []

    units = topics or list(sentences.values())
    for t in units:
        if any(f in (t.get("flags") or []) for f in ("production_cue", "offscreen", "backchannel")):
            continue
        thought = _span_from_seg(t)
        if not thought.text or len(thought.text.split()) < _MIN_THOUGHT_WORDS:
            continue
        children = [sentences[c] for c in (t.get("child_seg_ids") or []) if c in sentences]
        core = _span_from_seg(children[-1]) if children else thought
        out.append(Thought(
            speaker=t.get("speaker"),
            thought=thought, core=core, punch=core, setup=None,
            strength=0.6,
        ))
    return out


# --------------------------------------------------------------------------
# Compute (LLM with fallback)
# --------------------------------------------------------------------------

def compute_thoughts(file_id: str, *, llm: Optional[LLMClient] = None) -> List[Thought]:
    """Thoughts for one clip: the LLM pass when eligible, else the L1 fallback.

    Never raises -- any failure degrades to the deterministic fallback."""
    settings = get_settings()
    source = ss.load_source(file_id)
    words = _real_words(source) if source else []
    if not words:
        return []

    if not settings.enable_thought_segments:
        return segment_fallback(file_id)
    if len(words) > settings.thoughts_max_words:
        logger.info("thoughts: %s has %d words (> %d); using L1 fallback",
                    file_id, len(words), settings.thoughts_max_words)
        return segment_fallback(file_id)

    if llm is None:
        try:
            llm = get_llm(
                provider=(settings.thoughts_provider or settings.autoedit_provider or None),
                model=(settings.thoughts_model or settings.autoedit_model or None),
            )
        except Exception:
            logger.exception("thoughts: no LLM client; using L1 fallback for %s", file_id)
            return segment_fallback(file_id)

    try:
        thoughts = segment_with_llm(words, llm)
    except Exception:
        logger.exception("thoughts: LLM pass failed for %s; using L1 fallback", file_id)
        return segment_fallback(file_id)

    if not thoughts:
        logger.info("thoughts: LLM returned nothing for %s; using L1 fallback", file_id)
        return segment_fallback(file_id)
    return thoughts


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _ensure_table(conn) -> None:
    conn.execute(
        """
        create table if not exists speech_thoughts (
            file_id        uuid primary key,
            schema_version text not null,
            thoughts       jsonb not null,
            created_at     timestamptz not null default now()
        )
        """
    )


def _signature(words: List[dict]) -> str:
    """Cheap content signature: word count + schema version. A re-transcribe
    changes the count, so the cached row recomputes."""
    payload = json.dumps({"words": len(words), "sv": THOUGHT_SCHEMA_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def _get_row(conn, file_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "select schema_version, thoughts from speech_thoughts where file_id = %s",
        (file_id,),
    ).fetchone()
    if not row:
        return None
    thoughts = row[1] if isinstance(row[1], list) else json.loads(row[1])
    return {"schema_version": row[0], "thoughts": thoughts}


def _put_row(conn, file_id: str, sig: str, thoughts: List[Dict[str, Any]]) -> None:
    conn.execute(
        """
        insert into speech_thoughts (file_id, schema_version, thoughts)
        values (%s, %s, %s)
        on conflict (file_id) do update set
            schema_version = excluded.schema_version,
            thoughts = excluded.thoughts,
            created_at = now()
        """,
        (file_id, sig, json.dumps(thoughts)),
    )


def precompute_thoughts(file_id: str, *, llm: Optional[LLMClient] = None) -> int:
    """Compute + store thoughts for one file. Returns the count written."""
    source = ss.load_source(file_id)
    words = _real_words(source) if source else []
    if not words:
        return 0
    thoughts = [t.to_dict() for t in compute_thoughts(file_id, llm=llm)]
    with _pg_conn() as conn:
        _ensure_table(conn)
        _put_row(conn, file_id, _signature(words), thoughts)
    logger.info("thoughts: %s -> %d thoughts", file_id, len(thoughts))
    return len(thoughts)


def get_thoughts(file_id: str) -> List[Thought]:
    """Cached thoughts for one file, lazily computing + storing on a miss/stale
    signature. Fail-open: a cache error degrades to a live compute."""
    source = ss.load_source(file_id)
    words = _real_words(source) if source else []
    if not words:
        return []
    sig = _signature(words)
    try:
        with _pg_conn() as conn:
            _ensure_table(conn)
            hit = _get_row(conn, file_id)
            if hit and hit["schema_version"] == sig:
                return [Thought.from_dict(d) for d in hit["thoughts"]]
            thoughts = compute_thoughts(file_id)
            _put_row(conn, file_id, sig, [t.to_dict() for t in thoughts])
            return thoughts
    except Exception:
        logger.exception("thoughts: cache path failed for %s; live compute", file_id)
        return compute_thoughts(file_id)


# --------------------------------------------------------------------------
# Worker task (warms the cache post-L2, on the same l2 queue)
# --------------------------------------------------------------------------

def _register_task():
    """Lazily register the procrastinate task. Imported here (not at module top)
    so importing the pure logic for tests never pulls in the worker app."""
    from procrastinate import RetryStrategy

    from app.services.jobs import app

    @app.task(name="l3_precompute_thoughts", queue="l2",
              retry=RetryStrategy(max_attempts=2, exponential_wait=10))
    def _task(file_id: str) -> None:
        precompute_thoughts(file_id)

    return _task


def defer_thoughts(file_id: str) -> None:
    """Enqueue the thought pass on the l2 queue (where perception/hero precompute
    already run). Best-effort: warms the cache so reads don't pay the LLM cost."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        (enqueue_app.configure_task("l3_precompute_thoughts", queue="l2")
         .defer(file_id=file_id))
