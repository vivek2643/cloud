"""
Persistent file-based audit log for everything the AI brain does.

Why file-based and not Postgres-only:
- Survives uvicorn --reload restarts (Postgres rows do too, but they aren't
  human-grep-able when you just want to "see what happened").
- A single .json per event you can `cat`, diff, or attach to a bug report.
- Browsable from the frontend via /api/logs.

Layout
------
backend/logs/
    l1/
        <file_id>.json          # full L1 analysis dump per file (overwritten on re-run)
    edits/
        <iso_ts>__<short_id>.json   # one file per edit-request
    runtime.jsonl                   # append-only diary (info-level events)

All writes are best-effort: if the disk is full or the path is read-only,
the caller's flow is not interrupted.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Resolve <project>/backend/logs at import time. Caller can override with
# AUDIT_LOG_DIR env var if they want logs elsewhere (e.g. Docker volume).
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR = Path(os.environ.get("AUDIT_LOG_DIR", str(_DEFAULT_DIR)))

_lock = threading.Lock()


def _ensure(*subdirs: str) -> Path:
    p = LOG_DIR.joinpath(*subdirs)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _safe_write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write so partial writes never break readers."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(path)
    except Exception:
        logger.exception("audit_log: failed to write %s", path)


def _append_runtime(event: Dict[str, Any]) -> None:
    try:
        _ensure()
        with _lock, (LOG_DIR / "runtime.jsonl").open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        logger.exception("audit_log: failed to append runtime event")


# --- L1 analysis dumps ---------------------------------------------------

def write_l1_analysis(file_id: str, analysis: Dict[str, Any]) -> Path:
    """Dump the post-run L1 view of a file to logs/l1/<file_id>.json.

    `analysis` is expected to be already-serializable (no numpy arrays,
    no datetime). Use `build_l1_snapshot()` to construct one from the DB.
    """
    out = _ensure("l1") / f"{file_id}.json"
    payload = {"logged_at": _now_iso(), **analysis}
    _safe_write_json(out, payload)
    _append_runtime({"ts": _now_iso(), "kind": "l1_complete", "file_id": file_id, "log_path": str(out)})
    return out


# --- Edit-request dumps --------------------------------------------------

def open_edit_log(prompt: str) -> "EditLogContext":
    """Return a context object that an edit-request flow incrementally fills,
    flushed to disk on close. The id is short_uuid + iso_ts so the filename is
    sortable and easy to share."""
    short = uuid.uuid4().hex[:8]
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{ts}__{short}.json"
    out = _ensure("edits") / name
    return EditLogContext(out, prompt)


class EditLogContext:
    def __init__(self, path: Path, prompt: str):
        self.path = path
        self.data: Dict[str, Any] = {
            "logged_at": _now_iso(),
            "prompt": prompt,
            "status": "started",
            "stages": {},
        }
        self._flush()

    def stage(self, name: str, value: Any) -> None:
        self.data["stages"][name] = value
        self._flush()

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._flush()

    def fail(self, error: str) -> None:
        self.data["status"] = "failed"
        self.data["error"] = error
        self._flush()
        _append_runtime({"ts": _now_iso(), "kind": "edit_failed", "log_path": str(self.path), "error": error[:300]})

    def succeed(self) -> None:
        self.data["status"] = "ok"
        self._flush()
        _append_runtime({"ts": _now_iso(), "kind": "edit_ok", "log_path": str(self.path)})

    def _flush(self) -> None:
        _safe_write_json(self.path, self.data)


# --- Listing helpers (for /api/logs) -------------------------------------

def list_l1_logs() -> list[Dict[str, Any]]:
    p = LOG_DIR / "l1"
    if not p.exists():
        return []
    out: list[Dict[str, Any]] = []
    for f in sorted(p.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        out.append({
            "file_id": f.stem,
            "size_bytes": f.stat().st_size,
            "modified_at": _dt.datetime.fromtimestamp(f.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
        })
    return out


def list_edit_logs(limit: int = 100) -> list[Dict[str, Any]]:
    p = LOG_DIR / "edits"
    if not p.exists():
        return []
    out: list[Dict[str, Any]] = []
    for f in sorted(p.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            blob = json.loads(f.read_text())
        except Exception:
            continue
        out.append({
            "id": f.stem,
            "prompt": blob.get("prompt", ""),
            "status": blob.get("status", "unknown"),
            "duration_target_s": blob.get("stages", {}).get("query", {}).get("duration_target_s"),
            "actual_duration_s": blob.get("stages", {}).get("timeline_summary", {}).get("actual_duration_s"),
            "logged_at": blob.get("logged_at"),
        })
    return out


def read_edit_log(log_id: str) -> Optional[Dict[str, Any]]:
    p = LOG_DIR / "edits" / f"{log_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_l1_log(file_id: str) -> Optional[Dict[str, Any]]:
    p = LOG_DIR / "l1" / f"{file_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
