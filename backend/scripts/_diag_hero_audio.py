"""Read-only: verify the hero-audio / sync-group routing on the outlook cuts of
the fully-multicam project 57b689b3 (30 outlooks, 0 winners). Checks that each
outlook cut carries: a real sync_group_id, the group's authoritative (hero)
audio_file_id, a per-angle audio_offset_ms, and audio_align_confidence -- and
that the per-angle src windows are actually shifted by the offset (not clones).
Cross-checks audio_file_id against sync_groups.authoritative_audio_file_id.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import db  # noqa: E402

PID_PREFIX = "57b689b3"


def main():
    with db.connection() as conn:
        rid = conn.execute(
            "select id::text from ingest_runs where project_id::text like %s "
            "order by created_at desc limit 1", (PID_PREFIX + "%",),
        ).fetchone()[0]
        print("latest run:", rid)

        # 1. population of the audio-routing columns on outlook cuts
        row = conn.execute(
            """
            select count(*) as n,
                   count(*) filter (where sync_group_id is not null) as has_sync,
                   count(*) filter (where audio_file_id is not null) as has_audio,
                   count(*) filter (where audio_offset_ms is not null) as has_off,
                   count(*) filter (where audio_offset_ms <> 0) as nonzero_off,
                   count(*) filter (where audio_align_confidence is not null) as has_conf,
                   count(distinct sync_group_id) as n_groups,
                   count(distinct audio_file_id) as n_audiofiles,
                   count(distinct file_id) as n_anglefiles
              from cut_records
             where ingest_run_id=%s and kind='speech' and take_role='outlook'
            """,
            (rid,),
        ).fetchone()
        keys = ["outlooks", "sync_group_id set", "audio_file_id set", "audio_offset set",
                "audio_offset != 0", "confidence set", "distinct groups",
                "distinct audio files", "distinct angle files"]
        print("\n-- audio-routing column population on outlook cuts --")
        for k, v in zip(keys, row):
            print(f"   {k:24}: {v}")

        # 2. one beat's fan-out: angles + their shifted windows + offsets
        beat = conn.execute(
            """
            select take_group_id::text from cut_records
             where ingest_run_id=%s and kind='speech' and take_role='outlook'
             group by take_group_id order by count(*) desc limit 1
            """,
            (rid,),
        ).fetchone()[0]
        print(f"\n-- one take group ({beat[:8]}) fan-out across angles --")
        angles = conn.execute(
            """
            select file_id::text, src_in_ms, src_out_ms, audio_file_id::text,
                   audio_offset_ms, audio_align_confidence
              from cut_records
             where ingest_run_id=%s and take_group_id::text=%s and take_role='outlook'
             order by audio_offset_ms
            """,
            (rid, beat),
        ).fetchall()
        for f, a, b, af, off, conf in angles:
            print(f"   angle {f[:8]}  in={a:>7} out={b:>7}  hero_audio={af[:8]}  "
                  f"offset={off:>6}ms  conf={conf}")

        # 3. cross-check audio_file_id == sync group's authoritative audio file
        print("\n-- cross-check audio_file_id vs sync_groups.authoritative_audio_file_id --")
        chk = conn.execute(
            """
            select distinct c.sync_group_id::text, c.audio_file_id::text,
                   sg.authoritative_audio_file_id::text,
                   (c.audio_file_id::text = sg.authoritative_audio_file_id::text) as matches
              from cut_records c
              join sync_groups sg on sg.id::text = c.sync_group_id::text
             where c.ingest_run_id=%s and c.kind='speech' and c.take_role='outlook'
            """,
            (rid,),
        ).fetchall()
        for gid, af, auth, ok in chk:
            print(f"   group {gid[:8]}  cut.audio={af[:8]}  sg.auth={auth[:8]}  match={ok}")

        # 4. is the hero audio file real + does it carry audio?
        print("\n-- hero audio file(s) sanity --")
        for gid, af, auth, ok in chk:
            frow = conn.execute(
                "select filename, file_type, coalesce(duration_seconds,0) from files where id::text=%s",
                (af,),
            ).fetchone()
            print(f"   {af[:8]}: {frow}")


if __name__ == "__main__":
    main()
