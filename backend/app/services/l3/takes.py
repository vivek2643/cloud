"""
Take grouping: turn per-clip analysis into comparable "attempts" of the same
content, so the editor can pick the best one (or replace a part).

All deterministic, all post-VLM (no model calls):
  content unit  one span delivering one unit of content (from the VLM's
                `content_units`, or transcript sentences as a fallback).
  attempt       one delivery of a content unit. A unit splits into multiple
                attempts when the clip contains a retry (`restart_markers`).
  take group    all attempts that deliver the same spoken beat, matched by
                AUDIBLE SPEAKER + content-token OVERLAP across every clip in
                scope.

Matching is on the diarized speaker plus the set of content words, NOT a
whole-string ratio. Two cameras of the same line transcribe/segment it
differently (one cut bleeds into the next sentence, a few words differ), so a
full-string `SequenceMatcher` rejects genuine same-beat pairs. Token CONTAINMENT
(shared / smaller side) survives that drift; the speaker gate stops two
different people's "yeah exactly" from colliding.

Only groups with >= 2 attempts are returned -- the same beat captured more than
once. This module only finds *what is the same beat as what*; it does NOT crown
a "best" or tell the brain to drop anything -- quality judgment and the
placement decision are the brain's.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings

# Same-beat test: of the smaller cut's content tokens, this fraction must be
# shared with the other cut (containment, not whole-string similarity). Survives
# boundary drift -- a cut that bleeds into the next sentence still links to the
# tighter cut of the same line, because their shared run dominates the smaller.
CONTAINMENT_THRESHOLD = 0.6
# A match also needs at least this many SHARED content tokens, so a couple of
# common words ("the product", "you know") can't glue unrelated lines together.
MIN_SHARED_TOKENS = 4
# Below this many tokens a line is too generic to group on ("yeah", "thank
# you") -- it would create spurious cross-clip "takes". Such fragments are not
# meaningful take choices anyway.
MIN_KEY_TOKENS = 4
# Filler/stop tokens stripped before matching so "um, the the product" == "the product".
_FILLER_TOKENS = {"um", "uh", "er", "ah", "like", "you", "know", "sorry", "okay", "so"}
_WORD_RE = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class Attempt:
    attempt_id: str
    file_id: str
    unit_id: str
    start_ms: int
    end_ms: int
    kind: Optional[str]
    content_key: str       # normalized text (human/debug + token source)
    text: str              # human-readable
    speaker: Optional[str] = None       # dominant diarized (audible) speaker
    tokens: frozenset = frozenset()     # content tokens, for overlap matching
    is_restart: bool = False


@dataclass
class TakeGroup:
    group_id: str
    content_key: str
    attempts: List[Attempt] = field(default_factory=list)


# --------------------------------------------------------------------------
# Loading + normalization
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


def _as_doc(v: Any) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def normalize_key(text: Optional[str]) -> str:
    """Lower-case, drop fillers/punctuation, collapse whitespace -- the matching
    surface so the same line from two takes lands on the same key."""
    if not text:
        return ""
    toks = [t for t in _WORD_RE.findall(text.lower()) if t not in _FILLER_TOKENS]
    return " ".join(toks)


def _words_text(words: List[dict], start_ms: int, end_ms: int) -> str:
    out = [
        str(w.get("text", "")).strip()
        for w in words
        if not w.get("is_filler")
        and max(start_ms, int(w.get("start_ms", 0))) < min(end_ms, int(w.get("end_ms", 0)))
    ]
    return " ".join(t for t in out if t).strip()


def _span_speaker(words: List[dict], start_ms: int, end_ms: int) -> Optional[str]:
    """The diarized speaker who talks MOST inside the span (the audible speaker
    -- whose voice is on the track, regardless of who is on camera). ``None``
    when no diarized word falls in the span."""
    counts: Dict[str, int] = {}
    for w in words:
        if max(start_ms, int(w.get("start_ms", 0))) < min(end_ms, int(w.get("end_ms", 0))):
            spk = w.get("speaker")
            if spk:
                counts[spk] = counts.get(spk, 0) + 1
    return max(counts, key=counts.get) if counts else None


# --------------------------------------------------------------------------
# Per-clip attempts
# --------------------------------------------------------------------------

def _attempts_for_clip(
    file_id: str, perception: Optional[dict], segments: List[dict]
) -> List[Attempt]:
    words: List[dict] = []
    for seg in segments:
        words.extend(seg.get("words", []) or [])

    units: List[dict] = list((perception or {}).get("content_units") or [])
    restarts: List[dict] = list((perception or {}).get("restart_markers") or [])

    # Fallback: no VLM content units -> use transcript sentences as units.
    if not units:
        units = [
            {
                "unit_id": f"s{i}",
                "start_ms": int(seg.get("start_ms", 0)),
                "end_ms": int(seg.get("end_ms", 0)),
                "kind": "speech",
                "content_key": None,
            }
            for i, seg in enumerate(segments)
            if seg.get("text")
        ]

    out: List[Attempt] = []
    for u in units:
        u_id = str(u.get("unit_id", "u?"))
        u_start, u_end = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
        if u_end <= u_start:
            continue
        text = _words_text(words, u_start, u_end) or (u.get("label") or "")
        key = normalize_key(u.get("content_key") or text or u.get("label"))
        if len(key.split()) < MIN_KEY_TOKENS:
            continue  # too generic / empty to match reliably (e.g. "yeah", "thanks")

        # Split the unit at any restart that falls inside it: the moment before
        # the restart is the abandoned attempt, after it the retry. Both share
        # the unit's content_key.
        cuts = sorted(
            int(r.get("ms", 0))
            for r in restarts
            if u_start < int(r.get("ms", 0)) < u_end
            or str(r.get("restarts_unit") or "") == u_id
        )
        cuts = [c for c in cuts if u_start < c < u_end]
        bounds = [u_start, *cuts, u_end]
        for k in range(len(bounds) - 1):
            a_start, a_end = bounds[k], bounds[k + 1]
            if a_end - a_start < 200:
                continue
            seg_text = _words_text(words, a_start, a_end) or text
            out.append(
                Attempt(
                    attempt_id=f"{file_id[:8]}:{u_id}:{k}",
                    file_id=file_id,
                    unit_id=u_id,
                    start_ms=a_start,
                    end_ms=a_end,
                    kind=u.get("kind"),
                    content_key=key,
                    text=seg_text[:200],
                    speaker=_span_speaker(words, a_start, a_end),
                    tokens=frozenset(key.split()),
                    is_restart=k > 0,
                )
            )
    return out


# --------------------------------------------------------------------------
# Cross-clip clustering
# --------------------------------------------------------------------------

def _speaker_ok(a: Optional[str], b: Optional[str]) -> bool:
    """Speakers are compatible for grouping when they match, or when either is
    unknown (don't reject a real same-beat pair just because diarization was
    missing on one side)."""
    return (not a) or (not b) or (a == b)


def _containment(a: frozenset, b: frozenset) -> float:
    """Shared content tokens as a fraction of the SMALLER token set. 1.0 when one
    side's tokens are entirely inside the other -- which is exactly the boundary
    -drift case (a cut that bleeds into the next sentence vs the tighter cut)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cluster_attempts(attempts: List[Attempt]) -> List[TakeGroup]:
    """Greedy same-beat clustering on AUDIBLE SPEAKER + content-token CONTAINMENT.

    For each attempt we find the existing group whose seed shares the most of the
    smaller token set (>= CONTAINMENT_THRESHOLD and >= MIN_SHARED_TOKENS shared
    tokens) and whose speaker is compatible; else it starts a new group. Token
    containment (not whole-string ratio) is what links two cameras' differently
    -cut transcripts of the same line. Most lines are distinct, so this stays
    near-linear in practice (small token sets, early speaker/shared-count skips).
    """
    groups: List[TakeGroup] = []
    seeds: List[frozenset] = []        # seed token set, index-aligned with groups
    spk: List[Optional[str]] = []      # seed speaker, index-aligned with groups
    for att in attempts:
        toks = att.tokens
        if not toks:
            continue
        best: Optional[TakeGroup] = None
        best_score = CONTAINMENT_THRESHOLD
        for gi in range(len(groups)):
            if not _speaker_ok(att.speaker, spk[gi]):
                continue
            seed = seeds[gi]
            if len(toks & seed) < MIN_SHARED_TOKENS:
                continue
            score = _containment(toks, seed)
            if score >= best_score:
                best, best_score = groups[gi], score
        if best is None:
            groups.append(TakeGroup(group_id=f"tg{len(groups) + 1}",
                                    content_key=att.content_key, attempts=[att]))
            seeds.append(toks)
            spk.append(att.speaker)
        else:
            best.attempts.append(att)
    return groups


def build_take_groups(file_ids: List[str]) -> List[TakeGroup]:
    """All multi-attempt take groups across the clips in scope."""
    if not file_ids:
        return []
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, cp.perception, t.segments
              from files f
              left join clip_perception cp on cp.file_id = f.id
              left join transcripts t      on t.file_id  = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    all_attempts: List[Attempt] = []
    for fid, perception, segments in rows:
        all_attempts.extend(
            _attempts_for_clip(fid, _as_doc(perception), _as_list(segments))
        )

    groups = cluster_attempts(all_attempts)
    # Only groups where a real choice exists (>=2 attempts, from >=1 source).
    return [g for g in groups if len(g.attempts) >= 2]


# --------------------------------------------------------------------------
# Rendering for the orchestrator prompt
# --------------------------------------------------------------------------

def render_take_groups_text(groups: List[TakeGroup]) -> str:
    if not groups:
        return ""
    blocks: List[str] = [
        "SAME-BEAT GROUPS (the same spoken content captured more than once, as a "
        "retake or another camera angle -- judge each member by its text and "
        "quality and pick what fits):"
    ]
    for g in groups:
        head = f'  {g.group_id} ({len(g.attempts)} takes): "{g.attempts[0].text[:80]}"'
        blocks.append(head)
        for a in g.attempts:
            tag = " [retry]" if a.is_restart else ""
            blocks.append(
                f"    - {a.attempt_id}  clip {a.file_id} {a.start_ms}-{a.end_ms}ms{tag}"
            )
    return "\n".join(blocks)
