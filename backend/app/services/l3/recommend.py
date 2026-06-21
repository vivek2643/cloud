"""
Recommendations: an LLM filtration pass over a project's dialogue sentences.

One energy-independent text call judges each spoken sentence keep/drop ("is this
substantive, or is it filler / a false start / an off-topic aside?"). The verdict
lives on the sentence (`seg_id`), so the hero-cuts feed can flag a cut as
"recommended" at ANY energy band via the "contains a keeper" rule -- no per-band
call. The judgement is meaning-level (which lines are worth using), which is
exactly what the deterministic score can't see.

Provider/model route through the neutral ``get_llm`` factory, so this can run on
OpenAI while L3 stays on Anthropic, and switching is a config change.

Fails OPEN: any error (no key, bad JSON, provider down) yields an empty verdict,
which the feed treats as "recommend everything" -- the pool is never hidden by a
filtration hiccup.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.llm import get_llm, user_message

logger = logging.getLogger(__name__)

# Keys whose LLM filtration is currently being computed on a background thread,
# so the feed never blocks on the model and we never launch duplicate jobs.
_inflight: set = set()
_inflight_lock = threading.Lock()

# Sentences the L1 lexicon already flags as off-camera audio are not candidates
# (mirrors hero_cuts._lexicon_offcamera so the two pre-filters agree).
_OFFCAMERA_FLAGS = ("offscreen", "production_cue", "backchannel")

# verdict: member_key -> {"keep": bool, "reason": Optional[str]}
Verdict = Dict[str, Dict[str, object]]


def member_key(file_id: str, seg_id: str) -> str:
    """The project-global sentence key shared by the verdict and the hero cuts.
    Per-file seg_ids (sentence-N) repeat across clips, so we namespace by file."""
    return f"{file_id}:{seg_id}"


_SYSTEM = (
    "You are a sharp video editor curating which spoken lines from a shoot are "
    "worth keeping in a 'usable' pool. You will see, per clip, what it's about, "
    "then a numbered list of transcribed sentences.\n\n"
    "DROP a sentence only when it adds no editorial value on its own:\n"
    "  - pure filler / throat-clearing ('um', 'uh', 'so yeah', 'okay so').\n"
    "  - false starts / abandoned takes / 'wait, let me redo that'.\n"
    "  - off-topic asides or crew chatter unrelated to what the clip is about.\n"
    "  - redundant restatements that a stronger nearby line already covers.\n"
    "KEEP everything substantive: real points, answers, hooks, payoffs, lines a "
    "viewer would care about. When unsure, KEEP.\n\n"
    "Each sentence is prefixed with a numeric id in brackets, e.g. [42]. Refer to "
    "sentences ONLY by that number.\n"
    "Return ONLY JSON of this exact shape (no prose):\n"
    '{"drop": [{"id": <number>, "reason": "<short why>"}]}\n'
    "List ONLY the numbers to drop; everything else is kept implicitly."
)


def _pg_conn():
    import psycopg

    return psycopg.connect(get_settings().database_url, autocommit=True)


def _as_doc(v) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def _is_offcamera(seg: dict) -> bool:
    flags = seg.get("flags") or []
    return any(f in flags for f in _OFFCAMERA_FLAGS)


def _load_project(file_ids: List[str]) -> Tuple[List[dict], List[dict]]:
    """Return (per-clip context rows, candidate sentence rows) for the file set."""
    if not file_ids:
        return [], []
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, f.name, cp.perception, ds.segments
              from files f
              left join clip_perception   cp on cp.file_id = f.id
              left join dialogue_segments ds on ds.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    clips: List[dict] = []
    sentences: List[dict] = []
    for fid, name, perception, segments in rows:
        perc = _as_doc(perception) or {}
        clips.append(
            {
                "file_id": fid,
                "name": name,
                "content_type": perc.get("content_type"),
                "logline": perc.get("logline"),
                "topics": perc.get("topics") or [],
            }
        )
        seg_doc = _as_doc(segments) or {}
        for s in seg_doc.get("sentence", []) or []:
            text = (s.get("text") or "").strip()
            seg_id = s.get("seg_id")
            if not text or not seg_id or _is_offcamera(s):
                continue
            sentences.append(
                {
                    "seg_id": seg_id,
                    "file_id": fid,
                    # Globally-unique across the project: per-file seg_ids
                    # (sentence-0, sentence-1, ...) collide between clips, so the
                    # verdict / cuts MUST key on this, never bare seg_id.
                    "key": member_key(fid, seg_id),
                    "speaker": s.get("speaker"),
                    "text": text,
                }
            )
    return clips, sentences


def _build_prompt(clips: List[dict], sentences: List[dict]) -> str:
    """List every candidate sentence with a project-global numeric id (its index
    in `sentences`), grouped under clip headers. Indices are unambiguous across
    clips, unlike the per-file seg_ids."""
    name_by_id = {c["file_id"]: (c.get("name") or c["file_id"]) for c in clips}
    lines: List[str] = [f"PROJECT: {len(clips)} clip(s)."]
    for c in clips:
        bits = []
        if c.get("content_type"):
            bits.append(str(c["content_type"]))
        if c.get("logline"):
            bits.append(str(c["logline"]))
        if c.get("topics"):
            bits.append("topics: " + ", ".join(map(str, c["topics"][:6])))
        desc = " | ".join(bits) if bits else "(no perception summary)"
        lines.append(f"- {name_by_id[c['file_id']]}: {desc}")

    lines.append("")
    lines.append("SENTENCES — drop by the bracketed [number]:")
    cur: Optional[str] = None
    for i, s in enumerate(sentences):
        if s["file_id"] != cur:
            cur = s["file_id"]
            lines.append(f"\n## {name_by_id.get(cur, cur)}")
        spk = s.get("speaker") or "?"
        lines.append(f'[{i}] ({spk}) {s["text"]}')
    return "\n".join(lines)


def _extract_drop_ids(text: str, n: int) -> Dict[int, str]:
    """Parse the model's JSON and return {index: reason} for in-range indices."""
    if not text:
        return {}
    raw = text.strip()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            doc = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    out: Dict[int, str] = {}
    for item in (doc or {}).get("drop", []) or []:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if raw_id is None:
            raw_id = item.get("seg_id")
        try:
            idx = int(raw_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            out[idx] = str(item.get("reason") or "").strip()
    return out


def _tag(settings) -> str:
    """The provider/model/effort knobs that change the verdict -> part of the
    cache key, so flipping any of them recomputes instead of serving stale."""
    model = settings.recommend_model or ""
    return f"{settings.recommend_provider}:{model}:{settings.recommend_effort}"


def _cache_key(file_ids: List[str], sentences: List[dict], tag: str) -> str:
    keys = sorted(s["key"] for s in sentences)
    payload = json.dumps(
        {"files": sorted(file_ids), "segs": keys, "tag": tag},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _ensure_table(conn) -> None:
    conn.execute(
        """
        create table if not exists recommendations (
            key        text primary key,
            file_ids   jsonb not null,
            verdict    jsonb not null,
            model      text,
            created_at timestamptz not null default now()
        )
        """
    )


def _cache_get(key: str) -> Optional[Verdict]:
    try:
        with _pg_conn() as conn:
            _ensure_table(conn)
            row = conn.execute(
                "select verdict from recommendations where key = %s", (key,)
            ).fetchone()
        if row:
            return _as_doc(row[0]) or {}
    except Exception:  # cache is best-effort; never block the feed
        logger.exception("recommendations cache read failed")
    return None


def _cache_put(key: str, file_ids: List[str], verdict: Verdict, model: str) -> None:
    try:
        with _pg_conn() as conn:
            _ensure_table(conn)
            conn.execute(
                """
                insert into recommendations (key, file_ids, verdict, model)
                values (%s, %s, %s, %s)
                on conflict (key) do update set
                    verdict = excluded.verdict,
                    model = excluded.model,
                    created_at = now()
                """,
                (key, json.dumps(file_ids), json.dumps(verdict), model),
            )
    except Exception:
        logger.exception("recommendations cache write failed")


def build_recommendations(file_ids: List[str]) -> Verdict:
    """One LLM call: judge each candidate sentence keep/drop. Energy-independent.

    Returns {seg_id: {"keep": bool, "reason": Optional[str]}} for the dropped
    ids (kept sentences are simply absent -> treated as keep by callers). Empty
    on any failure or when disabled (fail-open).
    """
    settings = get_settings()
    if not settings.enable_recommendations:
        return {}

    clips, sentences = _load_project(file_ids)
    if not sentences:
        return {}

    key = _cache_key(file_ids, sentences, _tag(settings))

    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        client = get_llm(
            provider=settings.recommend_provider or None,
            model=settings.recommend_model or None,
        )
        resp = client.run(
            system=_SYSTEM,
            messages=[user_message(_build_prompt(clips, sentences))],
            max_tokens=settings.recommend_max_output_tokens,
            effort=settings.recommend_effort or None,
        )
        dropped = _extract_drop_ids(resp.text, len(sentences))
    except Exception:
        logger.exception("recommendations LLM call failed; recommending everything")
        return {}

    verdict: Verdict = {
        sentences[idx]["key"]: {"keep": False, "reason": reason or None}
        for idx, reason in dropped.items()
    }
    _cache_put(key, file_ids, verdict, client.model)
    logger.info(
        "recommendations: %d/%d sentences dropped across %d clip(s)",
        len(verdict), len(sentences), len(clips),
    )
    return verdict


def _spawn(file_ids: List[str], key: str) -> None:
    """Compute the verdict on a daemon thread so the feed request never blocks
    on the model. Guarded so concurrent feed loads don't fire duplicate calls."""
    with _inflight_lock:
        if key in _inflight:
            return
        _inflight.add(key)

    def _work():
        try:
            build_recommendations(file_ids)
        except Exception:
            logger.exception("background recommendations job failed")
        finally:
            with _inflight_lock:
                _inflight.discard(key)

    threading.Thread(target=_work, name="recommend", daemon=True).start()


def _get_or_trigger(file_ids: List[str]) -> Tuple[Verdict, bool]:
    """Return (verdict, ready). On a cache miss, kick off the LLM in the
    background and return ({}, False) immediately -- the feed shows everything
    until the picks land. Fail-open: any error yields ({}, True)."""
    settings = get_settings()
    if not settings.enable_recommendations:
        return {}, True
    try:
        _, sentences = _load_project(file_ids)
        if not sentences:
            return {}, True
        key = _cache_key(file_ids, sentences, _tag(settings))
        cached = _cache_get(key)
        if cached is not None:
            return cached, True
        _spawn(file_ids, key)
        return {}, False
    except Exception:
        logger.exception("get_recommendations failed; recommending everything")
        return {}, True


def get_recommendation_map(file_ids: List[str]) -> Verdict:
    """Verdict for the hero-cuts feed. Non-blocking; cached; fail-open."""
    return _get_or_trigger(file_ids)[0]


def get_recommendations(file_ids: List[str]) -> Tuple[Verdict, bool]:
    """(verdict, ready) for the API layer, so it can tell the client whether the
    LLM picks are final or still computing (so the UI can poll)."""
    return _get_or_trigger(file_ids)
