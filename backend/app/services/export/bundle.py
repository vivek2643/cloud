"""
export_options.plan.md Phase 4: the self-relinking ZIP bundle for the
"Rough cut for NLE" deliverable (B) -- .fcpxml + .srt + a relink manifest,
optionally the source media itself.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.export import fcpxml as fcpxml_mod
from app.services.export import srt as srt_mod
from app.services.processing import _download_from_r2, _upload_to_r2
from app.services.r2 import generate_presigned_get

EXPORT_PREFIX = "exports"

# Above this many bytes of TOTAL source media, skip copying media into the
# ZIP and write signed R2 GET URLs into manifest.json instead (Phase 4's
# "for very large projects" note) -- keeps the download a reasonable size
# even for a multi-hour, multi-clip project.
LARGE_PROJECT_MEDIA_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


@dataclass
class BundleFileEntry:
    """One source file, with everything bundle.py (and, via it, fcpxml.py)
    needs: the human filename (for the ZIP's relative media/ path and the
    FCPXML asset name, matched to each other 1:1 so the ZIP always
    self-relinks), the R2 key to fetch it from when media is included, its
    own byte size (for the large-project link-instead-of-copy decision),
    and its native pixel dimensions (fcpxml.py's framing math)."""
    file_id: str
    filename: str
    r2_key: str
    file_size_bytes: int = 0
    duration_ms: int = 0
    width: Optional[int] = None
    height: Optional[int] = None


def _fcpxml_lookup(file_lookup: Dict[str, BundleFileEntry]) -> Dict[str, fcpxml_mod.FcpxmlAsset]:
    return {
        fid: fcpxml_mod.FcpxmlAsset(
            file_id=fid, filename=e.filename, duration_ms=e.duration_ms, width=e.width, height=e.height,
        )
        for fid, e in file_lookup.items()
    }


def _safe_project_name(name: str) -> str:
    """A ZIP top-level folder / file basename must not carry path separators
    or other characters that could escape the intended directory or trip up
    a picky NLE importer -- keep it to a conservative, portable charset."""
    cleaned = "".join(c if (c.isalnum() or c in " _-") else "_" for c in (name or "Untitled")).strip()
    return cleaned or "Untitled"


def _referenced_file_ids(resolved: Dict[str, Any]) -> set:
    ids = {v.get("source_file_id") for v in (resolved.get("video_layers") or [])}
    ids |= {a.get("source_file_id") for a in (resolved.get("audio_layers") or [])}
    ids.discard(None)
    return ids


def build_rough_cut_bundle(
    resolved: Dict[str, Any],
    file_lookup: Dict[str, BundleFileEntry],
    *,
    project_name: str = "Untitled",
    include_media: bool = False,
    resolved_captions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the rough-cut ZIP (FCPXML + SRT + manifest.json + README,
    optionally media/) and upload it to R2, returning the output key.

      <ProjectName>/
        <ProjectName>.fcpxml
        <ProjectName>.srt
        manifest.json
        README.txt
        media/               (only if include_media, and under the size cap)
          <filename1>
          ...

    `include_media=False` (the common pro case) is a tiny, project-only
    bundle -- the editor relinks to their own local originals, dropped at
    `media/<filename>` next to the FCPXML (the FCPXML's own `<asset src=...>`
    already points there -- `manifest.json`'s `relpath` always names that
    same path, whether or not we actually populated it, so "drop your files
    here" is a literal, actionable instruction either way).

    `include_media=True` downloads each source from R2 into media/, UNLESS
    the project's total media size clears `LARGE_PROJECT_MEDIA_BYTES`, in
    which case the media/ copy is skipped and signed R2 GET URLs are written
    into manifest.json instead (never both -- a ZIP that silently omitted
    media without saying so would look broken, not intentionally link-based)."""
    safe_name = _safe_project_name(project_name)
    captions = resolved_captions if resolved_captions is not None else (resolved.get("captions") or [])

    fcpxml_text = fcpxml_mod.build_fcpxml(resolved, _fcpxml_lookup(file_lookup), project_name=safe_name)
    srt_text = srt_mod.build_srt(captions)

    referenced_ids = _referenced_file_ids(resolved)
    entries = [file_lookup[fid] for fid in referenced_ids if fid in file_lookup]
    total_bytes = sum(e.file_size_bytes for e in entries)
    copy_media = include_media and total_bytes <= LARGE_PROJECT_MEDIA_BYTES
    link_media = include_media and not copy_media

    assets_manifest: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="edso_export_bundle_") as tmp:
        root = os.path.join(tmp, safe_name)
        media_dir = os.path.join(root, "media")
        os.makedirs(root, exist_ok=True)
        if copy_media:
            os.makedirs(media_dir, exist_ok=True)

        for e in entries:
            download_url = None
            if copy_media:
                _download_from_r2(e.r2_key, os.path.join(media_dir, e.filename))
            elif link_media:
                download_url = generate_presigned_get(e.r2_key, expires_in=86400)
            assets_manifest.append({
                "file_id": e.file_id, "filename": e.filename,
                "relpath": f"media/{e.filename}", "download_url": download_url,
                "duration_ms": e.duration_ms,
            })

        manifest = {
            "project": safe_name,
            "frame_rate": fcpxml_mod.FPS,
            "assets": assets_manifest,
        }

        with open(os.path.join(root, f"{safe_name}.fcpxml"), "w", encoding="utf-8") as f:
            f.write(fcpxml_text)
        with open(os.path.join(root, f"{safe_name}.srt"), "w", encoding="utf-8") as f:
            f.write(srt_text)
        with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
            f.write(_readme_text(safe_name, copy_media, link_media))

        zip_local = os.path.join(tmp, f"{safe_name}.zip")
        _zip_dir_store_mode(root, zip_local, arc_root=safe_name)

        out_key = f"{EXPORT_PREFIX}/{uuid.uuid4().hex}.zip"
        _upload_to_r2(zip_local, out_key, "application/zip")
    return out_key


def _readme_text(project_name: str, media_included: bool, media_linked: bool) -> str:
    if media_included:
        return (
            f"{project_name} -- rough cut export\n\n"
            f"Open {project_name}.fcpxml in DaVinci Resolve or Premiere Pro. Media is\n"
            f"included in ./media -- it relinks automatically on import.\n\n"
            f"{project_name}.srt is the same subtitles as a plain-text sidecar, in case\n"
            f"your NLE's caption import wants a separate file.\n"
        )
    if media_linked:
        return (
            f"{project_name} -- rough cut export\n\n"
            f"Open {project_name}.fcpxml in DaVinci Resolve or Premiere Pro, then relink\n"
            f"media using the signed download URLs in manifest.json (this project was too\n"
            f"large to bundle media directly -- the links are valid for 24 hours).\n\n"
            f"{project_name}.srt is the same subtitles as a plain-text sidecar.\n"
        )
    return (
        f"{project_name} -- rough cut export\n\n"
        f"Open {project_name}.fcpxml in DaVinci Resolve or Premiere Pro, then relink to\n"
        f"your own local copies of the original source files -- drop them into a `media`\n"
        f"folder next to the .fcpxml (see manifest.json for the exact filenames this\n"
        f"timeline references) and the import will relink automatically.\n\n"
        f"{project_name}.srt is the same subtitles as a plain-text sidecar.\n"
    )


def _zip_dir_store_mode(src_dir: str, dst_zip: str, *, arc_root: str) -> None:
    """Zip `src_dir` (whose basename is `arc_root`) in STORE mode -- video is
    already compressed, re-compressing it would only cost CPU/time for zero
    size benefit (export_options.plan.md's own guardrail)."""
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for dirpath, _dirnames, filenames in os.walk(src_dir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                arcname = os.path.join(arc_root, os.path.relpath(full, src_dir))
                zf.write(full, arcname)


def presigned_url_for(out_key: str, expires_in: int = 86400) -> str:
    return generate_presigned_get(out_key, expires_in=expires_in)
