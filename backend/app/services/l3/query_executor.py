"""
Run a structured query against L1 data and return ranked candidate shots.

Inputs: structured query dict from query_parser.parse_prompt.
Output: list of CandidateShot ordered by score (descending).

Storage: direct psycopg connection because the Supabase REST client can't run
halfvec HNSW cosine queries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services.l1 import embeddings as emb_mod
from app.services.l1.pipeline import _vec_to_pg

logger = logging.getLogger(__name__)


@dataclass
class CandidateShot:
    shot_id: str
    file_id: str
    file_name: str
    file_r2_key: str
    file_r2_proxy_key: Optional[str]
    duration_seconds: Optional[float]
    shot_index: int
    start_ms: int
    end_ms: int
    score: float
    keyframe_r2_key: Optional[str]
    transcript_text: Optional[str] = None
    is_musical: Optional[bool] = None
    # L1.C multi-keyframe enrichment (nullable for shots indexed pre-mig 005)
    intra_shot_variance: Optional[float] = None
    peak_motion_ms: Optional[int] = None
    blur_min: Optional[float] = None


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


# Columns selected by the chat retrieval queries, in CandidateShot order.
# Kept as a constant so the "<= cap" and "bucketed" branches stay in sync.
_CHAT_SELECT_COLS = """
            s.id              as shot_id,
            s.shot_index      as shot_index,
            s.start_ms        as start_ms,
            s.end_ms          as end_ms,
            s.keyframe_r2_key as keyframe_r2_key,
            s.intra_shot_variance as intra_shot_variance,
            s.peak_motion_ms      as peak_motion_ms,
            s.blur_min            as blur_min,
            f.id              as file_id,
            f.name            as file_name,
            f.r2_key          as file_r2_key,
            f.r2_proxy_key    as file_r2_proxy_key,
            f.duration_seconds as duration_seconds,
            t.text            as transcript_text,
            af.is_musical     as is_musical
"""


def _hydrate(rows) -> List["CandidateShot"]:
    """Map raw dict rows (selecting _CHAT_SELECT_COLS) into CandidateShot."""
    return [
        CandidateShot(
            shot_id=str(r["shot_id"]),
            file_id=str(r["file_id"]),
            file_name=r["file_name"],
            file_r2_key=r["file_r2_key"],
            file_r2_proxy_key=r.get("file_r2_proxy_key"),
            duration_seconds=r.get("duration_seconds"),
            shot_index=r["shot_index"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            score=float(r.get("score") or 0.0),
            keyframe_r2_key=r.get("keyframe_r2_key"),
            transcript_text=r.get("transcript_text"),
            is_musical=r.get("is_musical"),
            intra_shot_variance=r.get("intra_shot_variance"),
            peak_motion_ms=r.get("peak_motion_ms"),
            blur_min=r.get("blur_min"),
        )
        for r in rows
    ]


def fetch_candidates_by_shot_ids(
    user_id: str,
    shot_ids: List[str],
    file_ids: Optional[List[str]] = None,
) -> List[CandidateShot]:
    """
    Hydrate full CandidateShot rows for a known set of shot ids, scoped to
    one user (defense-in-depth). Used by the chat endpoint so shots from
    a previous turn's timeline remain in the catalog Claude sees, even if
    the new SigLIP query wouldn't surface them.

    When ``file_ids`` is given, the returned set is additionally restricted
    to those files. We also DROP shots whose file isn't in the scope --
    this prevents stale prior-timeline shots from leaking back into the
    catalog after the user changes their selection.
    """
    if not shot_ids:
        return []
    sql = """
        select
            s.id              as shot_id,
            s.shot_index      as shot_index,
            s.start_ms        as start_ms,
            s.end_ms          as end_ms,
            s.keyframe_r2_key as keyframe_r2_key,
            s.intra_shot_variance as intra_shot_variance,
            s.peak_motion_ms      as peak_motion_ms,
            s.blur_min            as blur_min,
            f.id              as file_id,
            f.name            as file_name,
            f.r2_key          as file_r2_key,
            f.r2_proxy_key    as file_r2_proxy_key,
            f.duration_seconds as duration_seconds,
            t.text            as transcript_text,
            af.is_musical     as is_musical
        from shots s
        join files f on f.id = s.file_id
        left join transcripts t on t.file_id = f.id
        left join audio_features af on af.file_id = f.id
        where s.id = any(%s::uuid[])
          and f.user_id = %s
    """
    params: List[Any] = [shot_ids, user_id]
    if file_ids:
        sql += " and f.id = any(%s::uuid[])"
        params.append(file_ids)
    with _pg() as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    return [
        CandidateShot(
            shot_id=str(r["shot_id"]),
            file_id=str(r["file_id"]),
            file_name=r["file_name"],
            file_r2_key=r["file_r2_key"],
            file_r2_proxy_key=r.get("file_r2_proxy_key"),
            duration_seconds=r.get("duration_seconds"),
            shot_index=r["shot_index"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            score=1.0,  # no SigLIP score for direct lookups
            keyframe_r2_key=r.get("keyframe_r2_key"),
            transcript_text=r.get("transcript_text"),
            is_musical=r.get("is_musical"),
            intra_shot_variance=r.get("intra_shot_variance"),
            peak_motion_ms=r.get("peak_motion_ms"),
            blur_min=r.get("blur_min"),
        )
        for r in rows
    ]


def retrieve_top_k(
    user_id: str,
    prompt: str,
    folder_id: Optional[str] = None,
    k: int = 50,
    file_ids: Optional[List[str]] = None,
) -> List[CandidateShot]:
    """
    Cheap, filter-less SigLIP retrieval.

    Used by the smart edit path (claude_editor): we narrow the haystack
    from "every shot the user owns" down to the K most semantically
    similar shots so Claude doesn't have to reason over thousands of
    rows. The actual editorial decision (what to pick, in what order,
    where to trim) happens in claude_editor, not here.

    No L2 filters, no transcript filters, no sub-clip logic. Just pgvector.
    """
    if not prompt or not prompt.strip():
        return []
    text_vec = emb_mod.embed_text(prompt.strip())
    vec_pg = _vec_to_pg(text_vec)

    sql = """
        select
            s.id              as shot_id,
            s.shot_index      as shot_index,
            s.start_ms        as start_ms,
            s.end_ms          as end_ms,
            s.keyframe_r2_key as keyframe_r2_key,
            s.intra_shot_variance as intra_shot_variance,
            s.peak_motion_ms      as peak_motion_ms,
            s.blur_min            as blur_min,
            f.id              as file_id,
            f.name            as file_name,
            f.r2_key          as file_r2_key,
            f.r2_proxy_key    as file_r2_proxy_key,
            f.duration_seconds as duration_seconds,
            t.text            as transcript_text,
            af.is_musical     as is_musical,
            (1 - (se.embedding <=> %s::halfvec)) as score
        from shots s
        join files f on f.id = s.file_id
        join shot_embeddings se on se.shot_id = s.id
        left join transcripts t on t.file_id = f.id
        left join audio_features af on af.file_id = f.id
        where f.user_id = %s
          and f.l1_status = 'ready'
    """
    params: List[Any] = [vec_pg, user_id]
    if file_ids:
        # Explicit file selection takes precedence over folder scope.
        sql += " and f.id = any(%s::uuid[])"
        params.append(file_ids)
    elif folder_id:
        sql += " and f.folder_id = %s"
        params.append(folder_id)
    sql += " order by se.embedding <=> %s::halfvec asc limit %s"
    params.extend([vec_pg, k])

    with _pg() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    return [
        CandidateShot(
            shot_id=str(r["shot_id"]),
            file_id=str(r["file_id"]),
            file_name=r["file_name"],
            file_r2_key=r["file_r2_key"],
            file_r2_proxy_key=r.get("file_r2_proxy_key"),
            duration_seconds=r.get("duration_seconds"),
            shot_index=r["shot_index"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            score=float(r["score"] or 0.0),
            keyframe_r2_key=r.get("keyframe_r2_key"),
            transcript_text=r.get("transcript_text"),
            is_musical=r.get("is_musical"),
            intra_shot_variance=r.get("intra_shot_variance"),
            peak_motion_ms=r.get("peak_motion_ms"),
            blur_min=r.get("blur_min"),
        )
        for r in rows
    ]


def retrieve_for_chat(
    user_id: str,
    prompt: str,
    folder_id: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    cap: int = 120,
) -> List[CandidateShot]:
    """
    Retrieval tuned for the conversational editor.

    Unlike `retrieve_top_k` (pure SigLIP top-K, which biases the catalog toward
    whatever the prompt embeds near and shuffles footage out of order), this
    returns shots in CHRONOLOGICAL order so Claude can read the footage like a
    story instead of a bag of best-matching frames:

      * If the in-scope footage has <= cap shots, return ALL of them in
        chronological order (file creation order, then shot index). The editor
        sees the complete story; nothing is dropped.

      * If there are more than cap shots, stay chronological but DOWNSAMPLE:
        split each file into temporal buckets and keep the top `per_bucket`
        shots per bucket by a BALANCED blended score, then re-emit the kept
        shots in chronological order. This preserves coverage across the whole
        timeline instead of clustering on the prompt.

    The blended score spreads weight across modalities so the catalog is
    neither transcript-biased nor visual-biased:
        0.5 * visual (SigLIP cosine to the prompt)
      + 0.3 * transcript (full-text rank vs the prompt)
      + 0.2 * emotional_valence (L2)
      + 0.2 * motion_magnitude (clamped)
      + 0.1 * (not blurry)
    """
    prompt = (prompt or "").strip()

    # Scope filter shared by the count, file-count, and both retrieval branches.
    scope_sql = "where f.user_id = %s and f.l1_status = 'ready'"
    scope_params: List[Any] = [user_id]
    if file_ids:
        scope_sql += " and f.id = any(%s::uuid[])"
        scope_params.append(file_ids)
    elif folder_id:
        scope_sql += " and f.folder_id = %s"
        scope_params.append(folder_id)

    with _pg() as conn:
        cnt_row = conn.execute(
            "select count(*) as n from shots s join files f on f.id = s.file_id " + scope_sql,
            scope_params,
        ).fetchone()
        total = int((cnt_row or {}).get("n") or 0)
        if total == 0:
            return []

        # ---- Branch A: everything fits -> chronological, nothing dropped ----
        if total <= cap:
            sql = (
                "select" + _CHAT_SELECT_COLS +
                "from shots s\n"
                "join files f on f.id = s.file_id\n"
                "left join transcripts t on t.file_id = f.id\n"
                "left join audio_features af on af.file_id = f.id\n"
                + scope_sql + "\n"
                "order by f.created_at asc, f.id asc, s.shot_index asc"
            )
            rows = conn.execute(sql, scope_params).fetchall()
            return _hydrate(rows)

        # ---- Branch B: too many shots -> chronological bucketed downsample ----
        files_row = conn.execute(
            "select count(distinct f.id) as n from shots s join files f on f.id = s.file_id " + scope_sql,
            scope_params,
        ).fetchone()
        num_files = max(1, int((files_row or {}).get("n") or 1))

        BUCKETS = 10
        per_bucket = max(1, round(cap / (BUCKETS * num_files)))

        if not prompt:
            # No text to embed: fall back to a chronological, blur-aware sample.
            score_expr = (
                "0.2 * coalesce(s.emotional_valence, 0)\n"
                "      + 0.2 * least(coalesce(s.motion_magnitude, 0) / 10.0, 1.0)\n"
                "      + 0.1 * case when s.blur_min is null or s.blur_min >= 50 then 1 else 0 end"
            )
            score_params: List[Any] = []
        else:
            text_vec = emb_mod.embed_text(prompt)
            vec_pg = _vec_to_pg(text_vec)
            score_expr = (
                "0.5 * (1 - (se.embedding <=> %s::halfvec))\n"
                "      + 0.3 * coalesce(ts_rank(t.tsv, plainto_tsquery('simple', %s)), 0)\n"
                "      + 0.2 * coalesce(s.emotional_valence, 0)\n"
                "      + 0.2 * least(coalesce(s.motion_magnitude, 0) / 10.0, 1.0)\n"
                "      + 0.1 * case when s.blur_min is null or s.blur_min >= 50 then 1 else 0 end"
            )
            score_params = [vec_pg, prompt]

        # Two nested subqueries: inner q1 computes the score + temporal bucket,
        # outer q2 ranks within (file, bucket). Postgres can't nest a window
        # function (ntile) inside another window's PARTITION BY, hence the split.
        sql = (
            "select * from (\n"
            "  select q1.*,\n"
            "         row_number() over (partition by file_id, bucket order by score desc) as rn\n"
            "  from (\n"
            "    select" + _CHAT_SELECT_COLS + ",\n"
            "      f.created_at as created_at,\n"
            "      (" + score_expr + ") as score,\n"
            "      ntile(" + str(BUCKETS) + ") over (partition by s.file_id order by s.start_ms) as bucket\n"
            "    from shots s\n"
            "    join files f on f.id = s.file_id\n"
            "    join shot_embeddings se on se.shot_id = s.id\n"
            "    left join transcripts t on t.file_id = f.id\n"
            "    left join audio_features af on af.file_id = f.id\n"
            "    " + scope_sql + "\n"
            "  ) q1\n"
            ") q2\n"
            "where rn <= %s\n"
            "order by created_at asc, file_id asc, shot_index asc"
        )
        params: List[Any] = list(score_params) + list(scope_params) + [per_bucket]
        if prompt:
            conn.execute("SET LOCAL hnsw.ef_search = 100")
        rows = conn.execute(sql, params).fetchall()
        return _hydrate(rows)


def run_query(
    user_id: str,
    query: Dict[str, Any],
    folder_id: Optional[str] = None,
    limit: int = 200,
    raw_prompt: Optional[str] = None,
) -> List[CandidateShot]:
    """
    Execute a structured query against L1 data.

    Strategy:
    1. If `semantic_query` is set (or `raw_prompt` is provided as a fallback),
       embed it with SigLIP -> halfvec cosine distance gives a base score.
    2. Hard SQL filters from the structured query (narrative_role, valence,
       acoustic_tags, transcript keywords).
    3. SOFT score boosts/penalties using L2 fields, applied universally so
       L2 enrichment passively improves ranking even when the prompt parser
       doesn't add explicit L2 filters:
         + 0.10 * coalesce(emotional_valence, 0)   (mild positive-content bias)
         + 0.05 if narrative_role = 'payoff'        (mild "highlight" bias)
         - 0.10 if blur_min < 50                    (penalty for blurry shots)
       Hard filters always win; these boosts only reorder within the
       already-filtered candidate set.
    4. Limit to user's files; optionally scope to one folder.
    """
    mi = query["must_include"]
    me = query["must_exclude"]
    semantic_query: Optional[str] = mi.get("semantic_query")
    transcript_keywords: List[str] = mi.get("transcript_keywords") or []
    exclude_keywords: List[str] = me.get("transcript_keywords") or []
    narrative_role: Optional[str] = mi.get("narrative_role")
    require_acoustic: List[str] = mi.get("acoustic_tags") or []
    exclude_acoustic: List[str] = me.get("acoustic_tags") or []
    min_valence = mi.get("min_valence")
    max_valence = mi.get("max_valence")

    # (C) Fallback semantic query: if the parser returned null, embed the
    # raw user prompt so we still get SigLIP-based ranking. The parser is
    # supposed to always emit a non-null value but we don't trust it.
    if not semantic_query and raw_prompt and raw_prompt.strip():
        semantic_query = raw_prompt.strip()
        logger.info("query_executor: using raw_prompt as fallback semantic_query")

    # Base SELECT
    sql_parts = [
        """
        select
            s.id            as shot_id,
            s.shot_index    as shot_index,
            s.start_ms      as start_ms,
            s.end_ms        as end_ms,
            s.keyframe_r2_key as keyframe_r2_key,
            s.intra_shot_variance as intra_shot_variance,
            s.peak_motion_ms      as peak_motion_ms,
            s.blur_min            as blur_min,
            f.id            as file_id,
            f.name          as file_name,
            f.r2_key        as file_r2_key,
            f.r2_proxy_key  as file_r2_proxy_key,
            f.duration_seconds as duration_seconds,
            t.text          as transcript_text,
            af.is_musical   as is_musical
        """
    ]

    # Build the score expression: SigLIP cosine base + L2 soft boosts. The
    # boosts are always applied; for shots without L2 data they evaluate to
    # 0 (NULL-safe via COALESCE / IS NOT NULL), so old data isn't penalized.
    L2_SOFT_BOOST_SQL = (
        "+ coalesce(s.emotional_valence, 0) * 0.10"
        "+ case when s.narrative_role = 'payoff' then 0.05 else 0 end"
        "- case when s.blur_min is not null and s.blur_min < 50 then 0.10 else 0 end"
    )
    if semantic_query:
        sql_parts.append(f", (1 - (se.embedding <=> %s::halfvec)) {L2_SOFT_BOOST_SQL} as score")
    else:
        sql_parts.append(f", 0.5::float8 {L2_SOFT_BOOST_SQL} as score")

    sql_parts.append("""
        from shots s
        join files f on f.id = s.file_id
        left join shot_embeddings se on se.shot_id = s.id
        left join transcripts t on t.file_id = f.id
        left join audio_features af on af.file_id = f.id
        where f.user_id = %s
          and f.l1_status = 'ready'
    """)

    params: List[Any] = []
    if semantic_query:
        text_vec = emb_mod.embed_text(semantic_query)
        params.append(_vec_to_pg(text_vec))
    params.append(user_id)

    if folder_id:
        sql_parts.append("and f.folder_id = %s")
        params.append(folder_id)

    # Numeric filters
    if mi.get("min_focus_score") is not None:
        sql_parts.append("and s.focus_score >= %s")
        params.append(mi["min_focus_score"])
    if mi.get("max_motion_magnitude") is not None:
        sql_parts.append("and s.motion_magnitude <= %s")
        params.append(mi["max_motion_magnitude"])
    if mi.get("min_motion_magnitude") is not None:
        sql_parts.append("and s.motion_magnitude >= %s")
        params.append(mi["min_motion_magnitude"])

    # `transcript_keywords` is a SOFT preference: matching shots get a score
    # boost but non-matching shots aren't excluded. This avoids the brittle
    # AND-stacking failure mode (e.g. "trailer cut about the issue" where
    # payoff shots don't mention the keyword and the query returns nothing).
    # Hard exclusion (must_exclude) remains a hard filter.
    if exclude_keywords:
        sql_parts.append("and (t.tsv is null or not (t.tsv @@ plainto_tsquery('simple', %s)))")
        params.append(" ".join(exclude_keywords))

    # --- L2 filters (degrade gracefully when columns are still null) ---
    if narrative_role:
        sql_parts.append("and s.narrative_role = %s")
        params.append(narrative_role)
    if min_valence is not None:
        sql_parts.append("and s.emotional_valence >= %s")
        params.append(min_valence)
    if max_valence is not None:
        sql_parts.append("and s.emotional_valence <= %s")
        params.append(max_valence)
    if require_acoustic:
        # at least one of the requested tags must appear in audio_features.acoustic_tags
        sql_parts.append("and af.acoustic_tags && %s::text[]")
        params.append(require_acoustic)
    if exclude_acoustic:
        sql_parts.append("and (af.acoustic_tags is null or not (af.acoustic_tags && %s::text[]))")
        params.append(exclude_acoustic)

    # Order by the composite score (high to low) so L2 soft boosts/penalties
    # actually take effect. Falls back to source order when scores tie.
    sql_parts.append("order by score desc, s.file_id, s.shot_index")

    sql_parts.append("limit %s")
    params.append(limit)

    sql = "\n".join(sql_parts)

    with _pg() as conn:
        # Bump HNSW search effort for better recall
        if semantic_query:
            conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    return [
        CandidateShot(
            shot_id=str(r["shot_id"]),
            file_id=str(r["file_id"]),
            file_name=r["file_name"],
            file_r2_key=r["file_r2_key"],
            file_r2_proxy_key=r.get("file_r2_proxy_key"),
            duration_seconds=r.get("duration_seconds"),
            shot_index=r["shot_index"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            score=float(r["score"] or 0.0),
            keyframe_r2_key=r.get("keyframe_r2_key"),
            transcript_text=r.get("transcript_text"),
            is_musical=r.get("is_musical"),
            intra_shot_variance=r.get("intra_shot_variance"),
            peak_motion_ms=r.get("peak_motion_ms"),
            blur_min=r.get("blur_min"),
        )
        for r in rows
    ]
