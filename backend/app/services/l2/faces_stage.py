"""
L2 Stage B: Face detection + ReID clustering.

Pipeline per keyframe:
  1. SCRFD detector (insightface) -> face bboxes + 512-d ArcFace embeddings.
  2. For each face embedding, query characters table for nearest match per
     user. Cosine > NEW_CHARACTER_THRESHOLD -> reuse uuid; else insert new
     "Person_N" row.
  3. Aggregate per-shot character ids into shots.tracked_character_ids.

Insightface ships ONNX models; first run downloads weights to ~/.insightface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services.l1.pipeline import _vec_to_pg

logger = logging.getLogger(__name__)

# Cosine similarity threshold for matching a face to an existing character.
NEW_CHARACTER_THRESHOLD = 0.55


@dataclass
class FaceHit:
    embedding: np.ndarray  # (512,) float32, L2-normalized
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2


class _FaceEngine:
    _app = None

    @classmethod
    def get(cls):
        if cls._app is None:
            from insightface.app import FaceAnalysis
            logger.info("Loading insightface FaceAnalysis (SCRFD + ArcFace)...")
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            cls._app = app
            logger.info("Insightface ready.")
        return cls._app


def detect_faces(image_path: str) -> List[FaceHit]:
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return []
    face_app = _FaceEngine.get()
    faces = face_app.get(img)
    hits: List[FaceHit] = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        bbox = tuple(float(v) for v in f.bbox.tolist())  # type: ignore[assignment]
        hits.append(FaceHit(embedding=emb, bbox=bbox))  # type: ignore[arg-type]
    return hits


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _find_or_create_character(
    conn: psycopg.Connection,
    user_id: str,
    embedding: np.ndarray,
) -> str:
    """Match embedding against existing characters; return existing or new uuid."""
    vec_str = _vec_to_pg(embedding)
    conn.execute("SET LOCAL hnsw.ef_search = 100")
    cur = conn.execute(
        """
        select id, (1 - (embedding <=> %s::halfvec)) as sim
          from characters
         where user_id = %s
         order by embedding <=> %s::halfvec asc
         limit 1
        """,
        (vec_str, user_id, vec_str),
    )
    row = cur.fetchone()
    if row and float(row["sim"]) >= NEW_CHARACTER_THRESHOLD:
        return str(row["id"])

    # Generate next Person_N label per user
    cur = conn.execute(
        "select count(*) as n from characters where user_id = %s",
        (user_id,),
    )
    n = int(cur.fetchone()["n"]) + 1  # type: ignore[index]
    label = f"Person_{chr(64 + n) if n <= 26 else n}"  # Person_A..Z, then Person_27

    cur = conn.execute(
        """
        insert into characters (user_id, label, embedding)
        values (%s, %s, %s::halfvec)
        returning id
        """,
        (user_id, label, vec_str),
    )
    return str(cur.fetchone()["id"])  # type: ignore[index]


def enrich_shot_with_faces(
    user_id: str,
    shot_id: str,
    keyframe_paths: List[str],
) -> List[str]:
    """
    For every keyframe of one shot, detect faces and either match to existing
    characters or create new ones. Writes the resulting uuid list to
    shots.tracked_character_ids and returns it.
    """
    if not keyframe_paths:
        return []

    character_ids: Set[str] = set()
    with _pg() as conn:
        for path in keyframe_paths:
            for face in detect_faces(path):
                cid = _find_or_create_character(conn, user_id, face.embedding)
                character_ids.add(cid)

        # uuid[] needs explicit cast in psycopg
        ids_list = list(character_ids)
        conn.execute(
            "update shots set tracked_character_ids = %s::uuid[] where id = %s",
            (ids_list, shot_id),
        )
    return list(character_ids)
