"""
Content pass: give the orchestrator a survey of WHAT EVERY CLIP CONTAINS, plus
a pairwise map of which clips share content vs. cover distinct material.

Why this exists (the dropped-content bug): the catalog is a teaser (one logline
per clip), so nothing forced the model to see each clip's actual material before
building -- it could, and did, build a whole edit from 2 of 7 clips and silently
drop ~30 min of unique footage. The CONTENT MAP puts a compact, dialogue-grounded
digest of *every* clip in scope into the prompt so the survey is unavoidable.

The CONTENT OVERLAP index then tells the model, for each clip pair, whether they
cover the SAME content (candidate takes/angles -- keep one, or cut between them)
or DISTINCT content (sequence them). This is the signal that replaces the brittle
precomputed take/sync groups: it is a HINT the model reasons over, never an action.

All deterministic, post-VLM, no model calls and no video access. Reuses the L1
diarized turns (`diarize.load_turns`) and the L2 perception already in the DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from app.services.l3.catalog import load_perceptions
from app.services.l3.diarize import Turn, load_turns
from app.services.l3.takes import normalize_key

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Digest sizing -- scales DOWN as the clip count rises to stay in prompt budget
# --------------------------------------------------------------------------

def _gist_budget(n_clips: int) -> Tuple[int, int]:
    """(total gist chars per clip, max sampled turns) for `n_clips` in scope."""
    if n_clips <= 4:
        return 700, 8
    if n_clips <= 8:
        return 360, 5
    if n_clips <= 15:
        return 200, 3
    return 110, 2


def _mmss(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60:d}:{s % 60:02d}"


# --------------------------------------------------------------------------
# Per-clip content
# --------------------------------------------------------------------------

@dataclass
class ClipContent:
    file_id: str
    name: str
    duration_s: float
    has_transcript: bool
    logline: Optional[str]
    topics: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)   # durable appearance traits
    gist: str = ""                                     # sampled, prompt-facing
    # Carried for the overlap index (not rendered): full normalized speech and
    # the raw turns, so the pairwise map reuses this single load.
    norm_text: str = ""
    turns: List[Turn] = field(default_factory=list)


def _durable_traits(perception: Optional[dict]) -> List[str]:
    """One short appearance line per person -- NO voice/p-id label (those are
    clip-local; cross-clip identity is the model's inference on appearance)."""
    out: List[str] = []
    for p in (perception or {}).get("persons") or []:
        desc = p.get("canonical_description") or p.get("role")
        if desc:
            out.append(str(desc))
    return out


def _sampled_gist(turns: List[Turn], lines: List[str], budget: int, max_turns: int) -> str:
    """A few transcript turns sampled EVENLY across the clip (not just the head),
    so the breadth of a clip's content is visible -- the thing that distinguishes
    a true duplicate from a clip that merely starts the same way."""
    paired = [(t, ln) for t, ln in zip(turns, lines) if ln]
    if not paired:
        return ""
    if len(paired) <= max_turns:
        picked = paired
    else:
        step = (len(paired) - 1) / (max_turns - 1) if max_turns > 1 else 0
        idxs = sorted({round(i * step) for i in range(max_turns)})
        picked = [paired[i] for i in idxs]

    per = max(40, budget // max(1, len(picked)))
    bits: List[str] = []
    for (start, _end, _spk), line in picked:
        # `line` is "[start-end] SPK: text" -- keep the spoken text, retime to mm:ss.
        text = line.split(": ", 1)[1] if ": " in line else line
        text = text.strip()
        if len(text) > per:
            text = text[: per - 1].rstrip() + "\u2026"
        bits.append(f"[{_mmss(start)}] {text}")
    return "  ".join(bits)


def build_content_map(file_ids: List[str]) -> List[ClipContent]:
    """One ClipContent per clip in scope, in the given order. Loads the L1 turns
    and L2 perception that are already in the DB; no model calls."""
    if not file_ids:
        return []
    from app.services.l3.catalog import build_catalog

    catalog = {c.file_id: c for c in build_catalog(file_ids)}
    perceptions = load_perceptions(file_ids)
    budget, max_turns = _gist_budget(len(file_ids))

    out: List[ClipContent] = []
    for fid in file_ids:
        cat = catalog.get(fid)
        if cat is None:
            continue
        transcript_text, _spk, turns = load_turns(fid)
        lines = transcript_text.split("\n") if transcript_text else []
        has_tx = bool(turns)

        if has_tx:
            gist = _sampled_gist(turns, lines, budget, max_turns)
        else:
            gist = (cat.logline or "(silent / no transcript)")[:budget]

        out.append(ClipContent(
            file_id=fid,
            name=cat.name,
            duration_s=cat.duration_s,
            has_transcript=has_tx,
            logline=cat.logline,
            topics=list(cat.topics or []),
            people=_durable_traits(perceptions.get(fid)),
            gist=gist,
            norm_text=normalize_key(" ".join(ln.split(": ", 1)[1] for ln in lines if ": " in ln)),
            turns=turns,
        ))
    return out


def render_content_map_text(contents: List[ClipContent]) -> str:
    """Prompt-facing CONTENT MAP: a digest of EVERY clip so the survey is forced."""
    if not contents:
        return "CONTENT MAP: (no clips in scope)"
    blocks: List[str] = [
        "CONTENT MAP (what each clip actually contains -- you are seeing EVERY clip's "
        "content here; survey all of it and make sure each clip's unique material is "
        "considered before you finalize; do not build the whole edit from one or two clips):"
    ]
    for c in contents:
        head = f'CLIP {c.file_id} "{c.name}" ({c.duration_s:.0f}s):'
        blocks.append(head)
        if c.people:
            blocks.append("  people: " + "; ".join(c.people[:4]))
        if c.topics:
            blocks.append("  topics: " + ", ".join(str(t) for t in c.topics[:6]))
        if c.gist:
            label = "dialogue" if c.has_transcript else "gist"
            blocks.append(f"  {label}: {c.gist}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Pairwise content-overlap index (replaces precomputed take/sync groups)
# --------------------------------------------------------------------------

# A pair below this shared-content fraction is treated as distinct material and
# not reported (keeps the block to genuine candidates).
_OVERLAP_REPORT = 0.25
# At/above this fraction the two clips deliver essentially the same content
# (a take or a synced angle); below it they merely share a passage.
_OVERLAP_SAME = 0.55
# Cap the token sequence compared per clip so a long interview can't make the
# O(n*m) match blow up; the head+tail are the most diagnostic for overlap.
_MAX_TOKENS = 3500


@dataclass
class OverlapPair:
    file_a: str
    file_b: str
    score: float                 # shared-content fraction, 0..1
    same_content: bool
    span_a: Optional[Tuple[int, int]]   # ms in clip a (coarse hint)
    span_b: Optional[Tuple[int, int]]   # ms in clip b (coarse hint)


def _clamp_tokens(toks: List[str]) -> List[str]:
    if len(toks) <= _MAX_TOKENS:
        return toks
    half = _MAX_TOKENS // 2
    return toks[:half] + toks[-half:]


def _coarse_span(a: int, size: int, n_tokens: int, duration_s: float) -> Optional[Tuple[int, int]]:
    """Map a matched token block [a, a+size) to a coarse ms span by proportion
    of the clip's token stream -> duration. A HINT, not a frame-accurate range."""
    if n_tokens <= 0 or duration_s <= 0:
        return None
    dur_ms = int(duration_s * 1000)
    start = int(dur_ms * a / n_tokens)
    end = int(dur_ms * (a + size) / n_tokens)
    return (max(0, start), min(dur_ms, max(start + 1, end)))


def build_overlap_index(contents: List[ClipContent]) -> List[OverlapPair]:
    """Cheap text-based pairwise overlap over the clips that have transcripts.
    Returns only pairs above the report floor, strongest first. A HINT for the
    model (same-content candidates) -- never an action."""
    tx = [c for c in contents if c.norm_text]
    toks: Dict[str, List[str]] = {c.file_id: _clamp_tokens(c.norm_text.split()) for c in tx}
    pairs: List[OverlapPair] = []

    for i in range(len(tx)):
        ci = tx[i]
        ti = toks[ci.file_id]
        if len(ti) < 6:
            continue
        for j in range(i + 1, len(tx)):
            cj = tx[j]
            tj = toks[cj.file_id]
            if len(tj) < 6:
                continue
            sm = SequenceMatcher(None, ti, tj, autojunk=False)
            if sm.quick_ratio() < _OVERLAP_REPORT * 0.6:
                continue  # cheap prefilter before the O(n*m) block search
            blocks = sm.get_matching_blocks()
            matched = sum(b.size for b in blocks)
            denom = min(len(ti), len(tj)) or 1
            score = matched / denom
            if score < _OVERLAP_REPORT:
                continue
            big = max(blocks, key=lambda b: b.size, default=None)
            span_a = span_b = None
            if big and big.size:
                span_a = _coarse_span(big.a, big.size, len(ti), ci.duration_s)
                span_b = _coarse_span(big.b, big.size, len(tj), cj.duration_s)
            pairs.append(OverlapPair(
                file_a=ci.file_id, file_b=cj.file_id,
                score=round(score, 2), same_content=score >= _OVERLAP_SAME,
                span_a=span_a, span_b=span_b,
            ))

    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs


def render_overlap_text(pairs: List[OverlapPair]) -> str:
    """Prompt-facing CONTENT OVERLAP block."""
    if not pairs:
        return ("CONTENT OVERLAP: no clips share spoken content -- every clip covers "
                "DISTINCT material, so sequence them; none are duplicate takes.")
    blocks: List[str] = [
        "CONTENT OVERLAP (clips that share spoken content -- candidates for the SAME "
        "take/angle: keep one, or if they are simultaneous angles cut between them. "
        "Clips NOT paired here cover DISTINCT content -- sequence them, never treat as "
        "duplicates. This is a HINT to reason over, not an instruction: use align_clips to "
        "verify two are SIMULTANEOUS angles (vs. separate takes of the same line) and "
        "score_span to compare take quality before dropping anything):"
    ]
    for p in pairs:
        kind = "same content" if p.same_content else "partial overlap"
        span = ""
        if p.span_a and p.span_b:
            span = (f" (~{_mmss(p.span_a[0])}-{_mmss(p.span_a[1])} in {p.file_a[:8]}"
                    f" / {_mmss(p.span_b[0])}-{_mmss(p.span_b[1])} in {p.file_b[:8]})")
        blocks.append(f"  {p.file_a} <-> {p.file_b}: {p.score:.2f} {kind}{span}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Survey guardrail -- which clips' unique content went entirely unused
# --------------------------------------------------------------------------

def _used_file_ids(document: dict) -> set:
    """Every clip the current document actually draws on (spine + layer ops)."""
    used = set()
    for seg in document.get("timeline") or []:
        fid = seg.get("file_id") or seg.get("source_file_id")
        if fid:
            used.add(fid)
    for op in document.get("operations") or []:
        fid = op.get("source_file_id")
        if fid:
            used.add(fid)
    return used


def content_coverage(
    contents: List[ClipContent], pairs: List[OverlapPair], document: dict
) -> dict:
    """Report clips whose UNIQUE content is entirely unused -- i.e. unused AND
    not a same-content twin of a used clip (so dropping it really loses material).
    The guardrail against silently building from a subset of the footage."""
    used = _used_file_ids(document)
    # A clip is "covered by a twin" only when its same-content partner is used.
    same_twins: Dict[str, set] = {}
    for p in pairs:
        if not p.same_content:
            continue
        same_twins.setdefault(p.file_a, set()).add(p.file_b)
        same_twins.setdefault(p.file_b, set()).add(p.file_a)

    unique_unused: List[dict] = []
    for c in contents:
        if c.file_id in used:
            continue
        twins_used = bool(same_twins.get(c.file_id, set()) & used)
        if twins_used:
            continue  # its content survives via a used duplicate/angle
        unique_unused.append({
            "file_id": c.file_id,
            "name": c.name,
            "duration_s": round(c.duration_s, 1),
            "logline": c.logline,
        })

    return {
        "clips_in_scope": len(contents),
        "clips_used": len(used),
        "unique_unused_count": len(unique_unused),
        "unique_unused": unique_unused,
        "note": (
            "Every clip above covers content found in NO clip you used. Confirm each "
            "omission is intentional (off-topic / redundant / lower quality) before "
            "finalizing; otherwise add its material."
        ) if unique_unused else "All distinct content is represented in the edit.",
    }
