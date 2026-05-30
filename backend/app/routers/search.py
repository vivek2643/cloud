"""
Semantic search across the user's drive.

GET /api/search?q=<text>&folder_id=<optional>&limit=<int>

Encodes the query with SigLIP 2's text tower and runs an HNSW cosine search
against shot_embeddings, joined back to files for human-readable results.
Sets hnsw.ef_search=100 per query for better recall (default 40 is too low
per 2026 pgvector tuning).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg.rows import dict_row

from app.auth import get_current_user_id
from app.config import get_settings
from app.services.l1 import embeddings as emb_mod
from app.services.l1.pipeline import _vec_to_pg

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


@router.get("")
def search(
    q: str = Query(..., min_length=1, description="Natural-language search text"),
    folder_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    text_vec = emb_mod.embed_text(q)
    vec_str = _vec_to_pg(text_vec)

    params: List[Any] = [vec_str, user_id]
    sql = [
        """
        select
            s.id                as shot_id,
            s.shot_index        as shot_index,
            s.start_ms          as start_ms,
            s.end_ms            as end_ms,
            s.keyframe_r2_key   as keyframe_r2_key,
            f.id                as file_id,
            f.name              as file_name,
            f.duration_seconds  as duration_seconds,
            f.r2_thumbnail_key  as thumb_key,
            (1 - (se.embedding <=> %s::halfvec)) as score
          from shot_embeddings se
          join shots s on s.id = se.shot_id
          join files f on f.id = s.file_id
         where f.user_id = %s
           and f.l1_status = 'ready'
        """
    ]
    if folder_id:
        sql.append("and f.folder_id = %s")
        params.append(folder_id)

    sql.append("order by se.embedding <=> %s::halfvec asc")
    params.append(vec_str)
    sql.append("limit %s")
    params.append(limit)

    with _pg() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = conn.execute("\n".join(sql), params)
        rows = cur.fetchall()

    return {
        "query": q,
        "results": [
            {
                "shot_id": str(r["shot_id"]),
                "shot_index": r["shot_index"],
                "start_ms": r["start_ms"],
                "end_ms": r["end_ms"],
                "keyframe_r2_key": r.get("keyframe_r2_key"),
                "file_id": str(r["file_id"]),
                "file_name": r["file_name"],
                "duration_seconds": r.get("duration_seconds"),
                "thumb_key": r.get("thumb_key"),
                "score": float(r["score"] or 0.0),
            }
            for r in rows
        ],
    }
