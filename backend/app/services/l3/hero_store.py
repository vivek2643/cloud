"""
Hero-cuts precompute cache.

The per-file hero-cut assembly is the expensive part of the feed (fused field +
anchors + speech/beat/combined candidates). It is deterministic given a file's
L1/L2 artifacts and the energy band, so we compute it ONCE per file per band --
immediately after L2 perception finishes, on the same worker queue -- and store
it here. Reads (the feed routes and the auto-editor) then only do the cheap
cross-file finishing pass (``hero_cuts.assemble_cached``), so a level is always
ready instantly.

There are exactly five product LEVELS = the five energy bands at their centers
(``hero_cuts.BAND_ENERGIES``). A requested energy snaps to the nearest band.

Cache shape (one row per file per band):
    hero_cuts_cache(file_id, band, energy, source_version, cuts jsonb, ...)
``source_version`` is a cheap content signature (durations + segment/unit counts
+ a PARAMS_VERSION); a mismatch means the artifacts (or the cut logic) changed,
so the row is recomputed. Best-effort throughout: a cache miss falls back to a
live compute, so reads never fail because precompute hasn't run.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from procrastinate import RetryStrategy

from app.config import get_settings
from app.services.jobs import app
from app.services.l3 import hero_cuts as hc
from app.services.l3.energy import energy_band

logger = logging.getLogger(__name__)

_N_BANDS = len(hc.BAND_ENERGIES)


def _pg_conn():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _ensure_table(conn) -> None:
    conn.execute(
        """
        create table if not exists hero_cuts_cache (
            file_id        uuid not null,
            band           int  not null,
            energy         real not null,
            source_version text not null,
            cuts           jsonb not null,
            created_at     timestamptz not null default now(),
            primary key (file_id, band)
        )
        """
    )


# --------------------------------------------------------------------------
# Source signature (cheap, batched -- avoids loading the heavy grid arrays)
# --------------------------------------------------------------------------

def _signatures(conn, file_ids: List[str]) -> Dict[str, Optional[str]]:
    """file_id -> content signature (None if the file has no artifacts yet).

    Counts only (no big arrays pulled), so it's cheap to call on every read."""
    rows = conn.execute(
        """
        select f.id::text,
               coalesce(f.duration_seconds, 0),
               coalesce(jsonb_array_length(ds.segments->'sentence'), 0),
               coalesce(jsonb_array_length(ds.segments->'topic'), 0),
               coalesce(jsonb_array_length(cp.perception->'atoms'), 0),
               (md.file_id is not null),
               (af.file_id is not null)
          from files f
          left join dialogue_segments ds on ds.file_id = f.id
          left join clip_perception   cp on cp.file_id = f.id
          left join motion_dynamics   md on md.file_id = f.id
          left join audio_features    af on af.file_id = f.id
         where f.id = any(%s::uuid[])
        """,
        (file_ids,),
    ).fetchall()
    out: Dict[str, Optional[str]] = {fid: None for fid in file_ids}
    for fid, dur_s, n_sent, n_topic, n_atoms, has_motion, has_audio in rows:
        # A file with nothing usable (no dialogue and no detection atoms) has no
        # hero cuts yet; leave its signature None so we don't cache an empty row.
        if not n_sent and not n_atoms:
            continue
        payload = json.dumps({
            "dur": int(float(dur_s) * 1000), "sent": int(n_sent),
            "topic": int(n_topic), "atoms": int(n_atoms),
            "motion": bool(has_motion), "audio": bool(has_audio),
            "pv": hc.PARAMS_VERSION,
        }, sort_keys=True)
        out[fid] = hashlib.sha1(payload.encode()).hexdigest()
    return out


# --------------------------------------------------------------------------
# Cache get / put
# --------------------------------------------------------------------------

def _get_rows(conn, file_ids: List[str], band: int) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        "select file_id::text, source_version, cuts from hero_cuts_cache "
        "where band = %s and file_id = any(%s::uuid[])",
        (band, file_ids),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for fid, sv, cuts in rows:
        out[fid] = {"source_version": sv,
                    "cuts": cuts if isinstance(cuts, list) else json.loads(cuts)}
    return out


def _put_row(conn, file_id: str, band: int, energy: float,
             sig: str, cuts: List[Dict[str, Any]]) -> None:
    conn.execute(
        """
        insert into hero_cuts_cache (file_id, band, energy, source_version, cuts)
        values (%s, %s, %s, %s, %s)
        on conflict (file_id, band) do update set
            energy = excluded.energy,
            source_version = excluded.source_version,
            cuts = excluded.cuts,
            created_at = now()
        """,
        (file_id, band, energy, sig, json.dumps(cuts)),
    )


# --------------------------------------------------------------------------
# Precompute (called after L2; same worker queue)
# --------------------------------------------------------------------------

def precompute_file(file_id: str) -> int:
    """Compute and store all five bands for one file. Returns the number of
    bands written (0 when the file has no usable artifacts yet)."""
    with _pg_conn() as conn:
        _ensure_table(conn)
        sig = _signatures(conn, [file_id]).get(file_id)
        if sig is None:
            logger.info("hero precompute: %s has no artifacts yet; skipping", file_id)
            return 0
        existing = _get_rows(conn, [file_id], band=0).get(file_id)
        written = 0
        for band in range(_N_BANDS):
            energy = hc.band_energy(band)
            cuts = hc.compute_file_cache(file_id, energy)
            _put_row(conn, file_id, band, energy, sig, cuts)
            written += 1
    logger.info("hero precompute: %s -> %d bands (sig %s)", file_id, written, sig[:8])
    return written


@app.task(name="l3_precompute_hero_cuts", queue="l2",
          retry=RetryStrategy(max_attempts=2, exponential_wait=5))
def l3_precompute_hero_cuts(file_id: str) -> None:
    precompute_file(file_id)


def defer_precompute(file_id: str) -> None:
    """Enqueue the precompute on the l2 queue (same queue perception runs on)."""
    from procrastinate import App, PsycopgConnector
    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        (enqueue_app.configure_task("l3_precompute_hero_cuts", queue="l2")
         .defer(file_id=file_id))


# --------------------------------------------------------------------------
# Read path
# --------------------------------------------------------------------------

def get_hero_feed(
    file_ids: List[str], energy: float = 0.5,
    channels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """The ranked hero-cuts feed, served from the per-file precompute cache.

    Snaps ``energy`` to the nearest band, loads each file's cached cuts (lazily
    backfilling any that are missing or stale), then runs the cheap cross-file
    finishing pass. Drop-in replacement for ``hero_cuts.build_hero_cuts`` on the
    read side. Fail-open: any cache error degrades to a live full build."""
    if not file_ids:
        return []
    band = energy_band(energy)
    try:
        cached_by_file = _cached_or_backfill(file_ids, band)
    except Exception:
        logger.exception("hero feed: cache path failed; live build")
        return hc.build_hero_cuts(file_ids, hc.band_energy(band),
                                  channels=channels)
    return hc.assemble_cached(cached_by_file, file_ids, channels)


def _cached_or_backfill(file_ids: List[str], band: int) -> Dict[str, List[Dict[str, Any]]]:
    energy = hc.band_energy(band)
    with _pg_conn() as conn:
        _ensure_table(conn)
        sigs = _signatures(conn, file_ids)
        rows = _get_rows(conn, file_ids, band)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for fid in file_ids:
            sig = sigs.get(fid)
            if sig is None:
                out[fid] = []          # nothing usable in this file yet
                continue
            hit = rows.get(fid)
            if hit and hit["source_version"] == sig:
                out[fid] = hit["cuts"]
                continue
            # Miss or stale -> compute this one band now and store it. (The
            # other four bands are filled by the post-L2 precompute task.)
            cuts = hc.compute_file_cache(fid, energy)
            _put_row(conn, fid, band, energy, sig, cuts)
            out[fid] = cuts
    return out


# --------------------------------------------------------------------------
# Multi-band accessors (for the footage-map / moment-tree builder)
# --------------------------------------------------------------------------

def signatures_for(file_ids: List[str]) -> Dict[str, Optional[str]]:
    """Public content signature per file (None when the file has no usable
    artifacts yet). Used by downstream caches (e.g. the footage moment-tree) so
    they invalidate in lockstep with the hero-cut precompute."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        _ensure_table(conn)
        return _signatures(conn, file_ids)


def get_anchor_cuts(file_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """The single ANCHOR-band cut-set per file: ``{file_id: [cut, ...]}``.

    Each cut now owns its full zoom ladder, so the moment-tree needs only ONE
    band -- the balanced anchor (one complete thought per cut) -- and reads
    every other zoom off the rungs. Falls back per file to the nearest non-empty
    band (tight/calm/sharp/broad) so a clip with no balanced cuts for its
    modality still yields a tree. Replaces the old five-band geometric collapse.
    """
    if not file_ids:
        return {}
    bands = get_band_cuts(file_ids)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for fid in file_ids:
        by_band = bands.get(fid, {})
        out[fid] = next((by_band[b] for b in (2, 3, 1, 4, 0) if by_band.get(b)), [])
    return out


def get_band_cuts(file_ids: List[str]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Every band's PRE-stacking per-file cuts: ``{file_id: {band: [cut, ...]}}``.

    Mirrors the lazy backfill of :func:`get_hero_feed` but exposes all five
    bands for each file (no cross-file stacking) -- the multi-resolution input
    the moment-tree builder collapses into moments + variants. Fail-open per
    band/file: a compute error leaves that band empty rather than raising."""
    out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {fid: {} for fid in file_ids}
    if not file_ids:
        return out
    with _pg_conn() as conn:
        _ensure_table(conn)
        sigs = _signatures(conn, file_ids)
        for band in range(_N_BANDS):
            energy = hc.band_energy(band)
            rows = _get_rows(conn, file_ids, band)
            for fid in file_ids:
                sig = sigs.get(fid)
                if sig is None:
                    out[fid][band] = []
                    continue
                hit = rows.get(fid)
                if hit and hit["source_version"] == sig:
                    out[fid][band] = hit["cuts"]
                    continue
                try:
                    cuts = hc.compute_file_cache(fid, energy)
                    _put_row(conn, fid, band, energy, sig, cuts)
                except Exception:
                    logger.exception("band cuts: compute failed file=%s band=%s", fid, band)
                    cuts = []
                out[fid][band] = cuts
    return out
