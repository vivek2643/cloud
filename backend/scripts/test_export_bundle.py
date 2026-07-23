"""
Pure unit tests for the rough-cut ZIP bundler (``app.services.export.bundle``)
-- no real R2, no ffmpeg. `_download_from_r2`/`_upload_to_r2`/
`generate_presigned_get` are monkeypatched to local-filesystem stand-ins so
the test never touches the network.

Run:  .venv/bin/python scripts/test_export_bundle.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.export import bundle  # noqa: E402


def _resolved_fixture():
    return {
        "duration_ms": 2000,
        "aspect": "landscape",
        "video_layers": [
            {"layer_id": "v0", "source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
             "prog_start_ms": 0, "prog_end_ms": 2000, "z": 0, "kind": "spine", "transform": {}},
        ],
        "audio_layers": [
            {"source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
             "prog_start_ms": 0, "prog_end_ms": 2000, "gain_db": 0.0, "duck_db": 0.0},
        ],
        "captions": [
            {"prog_start_ms": 0, "prog_end_ms": 1000,
             "lines": [{"words": [{"text": "Hello.", "t_in_ms": 0, "t_out_ms": 1000, "emphasized": False}]}]},
        ],
    }


def _file_lookup():
    return {
        "f1": bundle.BundleFileEntry(
            file_id="f1", filename="clipA.mov", r2_key="uploads/f1/clipA.mov",
            file_size_bytes=1024, duration_ms=10000, width=1920, height=1080,
        ),
    }


class _Patcher:
    def __init__(self):
        self._orig = {}

    def set(self, obj, name, value):
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        for (obj, name), value in self._orig.items():
            setattr(obj, name, value)


def _with_captured_upload(fn):
    """Run `fn(patcher, captured)` with bundle's R2 calls monkeypatched to
    local-filesystem stand-ins; `captured["zip_path"]` holds a COPY of the
    uploaded zip (made before the source tempdir is cleaned up) once `fn`
    returns."""
    p = _Patcher()
    captured = {}
    persist_dir = tempfile.mkdtemp(prefix="edso_test_export_persist_")

    def fake_download(r2_key, dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"fake-video-bytes-for-" + r2_key.encode())

    def fake_upload(local_path, r2_key, content_type="application/octet-stream"):
        captured["r2_key"] = r2_key
        captured["content_type"] = content_type
        dst = os.path.join(persist_dir, os.path.basename(local_path))
        shutil.copy(local_path, dst)
        captured["zip_path"] = dst

    def fake_presign(key, expires_in=3600):
        return f"https://example.invalid/signed/{key}?expires={expires_in}"

    p.set(bundle, "_download_from_r2", fake_download)
    p.set(bundle, "_upload_to_r2", fake_upload)
    p.set(bundle, "generate_presigned_get", fake_presign)
    try:
        fn(p, captured)
    finally:
        p.restore()
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_bundle_project_only_has_no_media_dir():
    def run(_p, captured):
        out_key = bundle.build_rough_cut_bundle(
            _resolved_fixture(), _file_lookup(), project_name="MyReel", include_media=False,
        )
        assert out_key.startswith(bundle.EXPORT_PREFIX + "/") and out_key.endswith(".zip"), out_key
        assert captured["r2_key"] == out_key
        assert captured["content_type"] == "application/zip"
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            names = set(zf.namelist())
        assert "MyReel/MyReel.fcpxml" in names, names
        assert "MyReel/MyReel.srt" in names, names
        assert "MyReel/manifest.json" in names, names
        assert "MyReel/README.txt" in names, names
        assert not any(n.startswith("MyReel/media/") for n in names), names
        print("ok  test_bundle_project_only_has_no_media_dir")
    _with_captured_upload(run)


def test_bundle_include_media_copies_files_into_media_dir():
    def run(_p, captured):
        bundle.build_rough_cut_bundle(
            _resolved_fixture(), _file_lookup(), project_name="MyReel", include_media=True,
        )
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            names = set(zf.namelist())
            data = zf.read("MyReel/media/clipA.mov")
        assert "MyReel/media/clipA.mov" in names, names
        assert data == b"fake-video-bytes-for-uploads/f1/clipA.mov"
        print("ok  test_bundle_include_media_copies_files_into_media_dir")
    _with_captured_upload(run)


def test_bundle_large_project_links_instead_of_copies():
    def run(_p, captured):
        orig_threshold = bundle.LARGE_PROJECT_MEDIA_BYTES
        bundle.LARGE_PROJECT_MEDIA_BYTES = 1  # force the "too large" branch
        try:
            bundle.build_rough_cut_bundle(
                _resolved_fixture(), _file_lookup(), project_name="MyReel", include_media=True,
            )
        finally:
            bundle.LARGE_PROJECT_MEDIA_BYTES = orig_threshold
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("MyReel/manifest.json"))
        assert not any(n.startswith("MyReel/media/") for n in names), names
        asset = manifest["assets"][0]
        assert asset["download_url"] is not None and asset["download_url"].startswith("https://"), asset
        assert asset["relpath"] == "media/clipA.mov", asset
        print("ok  test_bundle_large_project_links_instead_of_copies")
    _with_captured_upload(run)


def test_bundle_manifest_is_well_formed_and_matches_shape():
    def run(_p, captured):
        bundle.build_rough_cut_bundle(
            _resolved_fixture(), _file_lookup(), project_name="MyReel", include_media=False,
        )
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            manifest = json.loads(zf.read("MyReel/manifest.json"))
        assert manifest["project"] == "MyReel"
        assert manifest["frame_rate"] == 30
        assert len(manifest["assets"]) == 1
        asset = manifest["assets"][0]
        assert asset["file_id"] == "f1"
        assert asset["filename"] == "clipA.mov"
        assert asset["relpath"] == "media/clipA.mov"
        assert asset["download_url"] is None
        assert asset["duration_ms"] == 10000
        print("ok  test_bundle_manifest_is_well_formed_and_matches_shape")
    _with_captured_upload(run)


def test_bundle_zip_uses_store_mode_no_compression():
    def run(_p, captured):
        bundle.build_rough_cut_bundle(
            _resolved_fixture(), _file_lookup(), project_name="MyReel", include_media=True,
        )
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, (info.filename, info.compress_type)
        print("ok  test_bundle_zip_uses_store_mode_no_compression")
    _with_captured_upload(run)


def test_bundle_sanitizes_project_name_for_zip_paths():
    def run(_p, captured):
        bundle.build_rough_cut_bundle(
            _resolved_fixture(), _file_lookup(), project_name="My/Reel: Take 2?", include_media=False,
        )
        with zipfile.ZipFile(captured["zip_path"]) as zf:
            names = zf.namelist()
        # Exactly one top-level folder, and it carries none of the raw name's
        # path-hostile characters (a literal "/" or ":" in the folder name
        # would either break the archive layout or a picky NLE's import).
        top_level = {n.split("/")[0] for n in names}
        assert len(top_level) == 1, top_level
        folder = next(iter(top_level))
        assert "/" not in folder and ":" not in folder and "?" not in folder, folder
        assert any(n.endswith(".fcpxml") for n in names), names
        print("ok  test_bundle_sanitizes_project_name_for_zip_paths")
    _with_captured_upload(run)


def main():
    test_bundle_project_only_has_no_media_dir()
    test_bundle_include_media_copies_files_into_media_dir()
    test_bundle_large_project_links_instead_of_copies()
    test_bundle_manifest_is_well_formed_and_matches_shape()
    test_bundle_zip_uses_store_mode_no_compression()
    test_bundle_sanitizes_project_name_for_zip_paths()
    print("\nall export_bundle tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
