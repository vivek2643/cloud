"""
Layer B (edit-time): pick the keyframes to actually SHOW the editor.

Given the candidate shots for an edit and the user's brief, choose a budgeted,
visually-diverse, relevance-ranked set of keyframes to attach to the multimodal
editor call. This is the productionized version of scripts/eval_keyframe_select.

Reads the adaptive shot_keyframes set when present (migration 008) and falls
back to the legacy per-shot anchor embedding otherwise. Pure read-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import psycopg

from app.config import get_settings
from app.services.l1 import embeddings as emb_mod

logger = logging.getLogger(__name__)


@dataclass
class SelectedFrame:
    shot_id: str
    r2_key: str
    ts_ms: int
    kind: str
    score: float


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _parse_halfvec(raw) -> Optional[np.ndarray]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        v = np.asarray(raw, dtype=np.float32)
    else:
        s = str(raw).strip().lstrip("[").rstrip("]")
        if not s:
            return None
        v = np.fromstring(s, sep=",", dtype=np.float32)
    if v.size == 0:
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else v


@dataclass
class _Cand:
    shot_id: str
    r2_key: str
    ts_ms: int
    kind: str
    vec: np.ndarray


def _load_candidates(shot_ids: List[str]) -> List[_Cand]:
    """Adaptive frames from shot_keyframes; fall back to the legacy anchor."""
    out: List[_Cand] = []
    with _pg() as conn:
        try:
            rows = conn.execute(
                """
                select sk.shot_id, sk.kind, sk.ts_ms, sk.r2_key, sk.embedding::text
                  from shot_keyframes sk
                 where sk.shot_id = any(%s::uuid[]) and sk.embedding is not null
                 order by sk.shot_id, sk.frame_index
                """,
                (shot_ids,),
            ).fetchall()
        except psycopg.errors.UndefinedTable:
            rows = []
        for sid, kind, ts, key, emb in rows:
            vec = _parse_halfvec(emb)
            if vec is not None and key:
                out.append(_Cand(str(sid), key, int(ts or 0), kind, vec))

        covered = {c.shot_id for c in out}
        missing = [s for s in shot_ids if s not in covered]
        if missing:
            legacy = conn.execute(
                """
                select s.id, s.keyframe_r2_key, s.start_ms, s.end_ms,
                       se.embedding::text
                  from shots s
                  join shot_embeddings se on se.shot_id = s.id
                 where s.id = any(%s::uuid[]) and s.keyframe_r2_key is not null
                """,
                (missing,),
            ).fetchall()
            for sid, key, sms, ems, emb in legacy:
                vec = _parse_halfvec(emb)
                if vec is not None and key:
                    out.append(_Cand(str(sid), key, (int(sms) + int(ems)) // 2, "anchor", vec))
    return out


def select_frames_for_edit(
    shot_ids: List[str],
    brief: str,
    budget: int,
    per_shot_max: int = 1,
    lam: float = 0.6,
) -> List[SelectedFrame]:
    """MMR pick: relevance to the brief (SigLIP text<->image) traded off against
    visual diversity, capped per shot and by total budget. Returns [] on any
    failure so the caller can fall back to a text-only edit."""
    if budget <= 0 or not shot_ids:
        return []
    try:
        cands = _load_candidates(shot_ids)
        if not cands:
            return []
        mat = np.stack([c.vec for c in cands])

        brief = (brief or "").strip()
        if brief:
            qv = emb_mod.embed_text(brief)
            qv = qv / (np.linalg.norm(qv) + 1e-8)
            rel = mat @ qv
            eff_lam = lam
        else:
            rel = np.zeros(len(cands), dtype=np.float32)
            eff_lam = 0.0

        selected: List[int] = []
        shot_counts: dict[str, int] = {}
        remaining = set(range(len(cands)))
        max_sim = np.zeros(len(cands), dtype=np.float32)
        while remaining and len(selected) < budget:
            best_i, best = -1, -1e9
            for i in remaining:
                if shot_counts.get(cands[i].shot_id, 0) >= per_shot_max:
                    continue
                score = eff_lam * float(rel[i]) - (1.0 - eff_lam) * float(max_sim[i])
                if score > best:
                    best, best_i = score, i
            if best_i < 0:
                break
            selected.append(best_i)
            remaining.discard(best_i)
            shot_counts[cands[best_i].shot_id] = shot_counts.get(cands[best_i].shot_id, 0) + 1
            max_sim = np.maximum(max_sim, mat @ mat[best_i])

        return [
            SelectedFrame(
                shot_id=cands[i].shot_id, r2_key=cands[i].r2_key,
                ts_ms=cands[i].ts_ms, kind=cands[i].kind, score=float(rel[i]),
            )
            for i in selected
        ]
    except Exception:
        logger.exception("Layer B frame selection failed; editor will run text-only")
        return []
