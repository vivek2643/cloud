"""color_scene_grouping.plan.md Phase 1: join each grade shot's
(file_id, in_ms..out_ms) to the cut_record it was cut from, so the real
scene metadata (already computed at ingest, but never carried onto the
timeline seg) is available for grouping. Pure lookup + max-overlap join;
best-effort and fail-open (no covering run / no overlapping cut -> that
shot simply gets empty metadata and falls back to the RGB base)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ShotCutMeta:
    """Real scene metadata for one shot, joined from its covering cut_record."""
    speaker_person: Optional[str] = None
    on_camera: Optional[bool] = None
    label: str = ""
    summary: str = ""
    voice_ids: List[str] = field(default_factory=list)
    take_group_id: Optional[str] = None
    sync_group_id: Optional[str] = None


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def lookup_shot_cut_meta(
    shots: List[Tuple[str, str, int, int]],   # (key, file_id, in_ms, out_ms)
) -> Dict[str, ShotCutMeta]:
    """key -> ShotCutMeta for every shot with a covering cut_record. Best-
    effort: any DB error, no covering run, or no overlapping cut yields an
    empty dict entry-free result for that shot (caller treats missing as
    'no metadata')."""
    out: Dict[str, ShotCutMeta] = {}
    file_ids = sorted({fid for _, fid, _, _ in shots})
    if not file_ids:
        return out
    try:
        from app.services.l3 import cuts_v3_read

        run_id = cuts_v3_read.latest_run_for_files(file_ids)
        if run_id is None:
            return out
        rows = cuts_v3_read.rows_for_run(run_id, file_ids)
    except Exception:
        return out   # fail-open: grouping falls back to the RGB base

    by_file: Dict[str, List[dict]] = {}
    for r in rows:
        by_file.setdefault(r["file_id"], []).append(r)

    for key, fid, in_ms, out_ms in shots:
        best, best_ov = None, 0
        for r in by_file.get(fid, []):
            ov = _overlap(in_ms, out_ms, int(r["src_in_ms"]), int(r["src_out_ms"]))
            if ov > best_ov:
                best, best_ov = r, ov
        if best is None or best_ov <= 0:
            continue
        out[key] = ShotCutMeta(
            speaker_person=best.get("speaker_person"),
            on_camera=best.get("on_camera"),
            label=str(best.get("label") or ""),
            summary=str(best.get("summary") or ""),
            voice_ids=list(best.get("voice_ids") or []),
            take_group_id=best.get("take_group_id"),
            sync_group_id=best.get("sync_group_id"),
        )
    return out
