"""Read-only: replay the new per-beat energy noise gate over EXISTING speech
cut_records to confirm it drops hallucination-over-noise cuts without nuking
real speech. For every kind='speech' cut in each project's latest run, compute
the beat's median rms_db over its window and the file's voiced/floor refs
(identical logic to inputs._energy_refs + delivery.is_voiced_beat), then report
how many would be dropped and show samples with their label/duration/energy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import db  # noqa: E402
from app.services.vcut.speech.delivery import _median, _rms_slice, is_voiced_beat  # noqa: E402
from app.services.vcut.speech.inputs import _energy_refs  # noqa: E402

_rms_cache = {}


def _rms(conn, file_id):
    if file_id not in _rms_cache:
        row = conn.execute(
            "select rms_db, prosody_hop_ms from audio_features where file_id=%s", (file_id,)
        ).fetchone()
        if not row:
            _rms_cache[file_id] = ([], 0, 0.0, 0.0)
        else:
            rms = list(row[0] or [])
            voiced, floor = _energy_refs(rms)
            _rms_cache[file_id] = (rms, int(row[1] or 0), voiced, floor)
    return _rms_cache[file_id]


def main():
    with db.connection() as conn:
        runs = conn.execute(
            "select distinct on (project_id) id::text from ingest_runs order by project_id, created_at desc"
        ).fetchall()
        run_ids = [r[0] for r in runs]
        cuts = conn.execute(
            """
            select file_id::text, src_in_ms, src_out_ms, coalesce(label,''), coalesce(summary,'')
              from cut_records
             where kind='speech' and ingest_run_id = any(%s)
            """,
            (run_ids,),
        ).fetchall()

        kept = dropped = no_rms = 0
        drop_samples, keep_samples = [], []
        for fid, a, b, label, summary in cuts:
            rms, hop, voiced, floor = _rms(conn, fid)
            if not rms or hop <= 0:
                no_rms += 1
                continue
            energy = _median(_rms_slice(rms, hop, a, b))
            ok = is_voiced_beat(energy, voiced, floor)
            dur = (b - a) / 1000.0
            rec = (label[:40], dur, energy, floor, voiced, summary[:50])
            if ok:
                kept += 1
                if len(keep_samples) < 8:
                    keep_samples.append(rec)
            else:
                dropped += 1
                drop_samples.append(rec)

        total = kept + dropped
        print(f"speech cuts scored: {total}  (no rms data, skipped: {no_rms})")
        print(f"  KEEP (real speech): {kept}")
        print(f"  DROP (gated noise): {dropped}  "
              f"({(100.0*dropped/total if total else 0):.1f}%)\n")

        print("-- sample DROPPED cuts (label | dur | energy floor voiced | summary) --")
        for label, dur, e, fl, vo, summ in drop_samples[:25]:
            print(f"   {label:40} {dur:5.1f}s  e={e:6.1f} fl={fl:6.1f} vo={vo:6.1f}  {summ}")

        print("\n-- sample KEPT cuts for contrast --")
        for label, dur, e, fl, vo, summ in keep_samples:
            print(f"   {label:40} {dur:5.1f}s  e={e:6.1f} fl={fl:6.1f} vo={vo:6.1f}  {summ}")


if __name__ == "__main__":
    main()
