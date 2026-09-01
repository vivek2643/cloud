#!/usr/bin/env python
"""
Manage private-beta invite codes (see migrations/058, 059).

    python scripts/invite_codes.py list
    python scripts/invite_codes.py new --uses 5
    python scripts/invite_codes.py new -n 10 --uses 5 --note "launch batch"
    python scripts/invite_codes.py new --uses 20 --expires 14
    python scripts/invite_codes.py revoke EDSO-1A2B

Run from backend/ -- Settings reads ../.env relative to this package, so
running from the repo root silently loses the credentials.

A "seat" is one browser that successfully entered the code at /invite, not one
account: the gate runs before an account exists, so there is nothing to tie a
redemption to. The same person on a phone and a laptop spends two seats.
"""
import argparse
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402


def _connect():
    settings = get_settings()
    return psycopg.connect(
        settings.database_url, autocommit=True, **settings.pg_connect_kwargs()
    )


def _generate() -> str:
    # Matches the existing batch's shape (EDSO-01D2). Uppercase hex avoids the
    # letter/digit lookalikes that make codes painful to read down a phone.
    return f"EDSO-{secrets.token_hex(2).upper()}"


def cmd_new(conn, args: argparse.Namespace) -> None:
    made: list[str] = []
    # Retry on collision rather than trusting 16 bits of randomness to be
    # unique; `on conflict do nothing` reports the miss via rowcount.
    attempts = 0
    while len(made) < args.number and attempts < args.number * 20:
        attempts += 1
        code = _generate()
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.invite_codes (code, max_uses, note, expires_at)
                values (%s, %s, %s,
                        -- Explicit ::int both times: with no --expires the
                        -- parameter arrives as a bare NULL and Postgres has no
                        -- other context to infer a type from.
                        case when %s::int is null then null
                             else now() + make_interval(days => %s::int) end)
                on conflict (code) do nothing
                """,
                (code, args.uses, args.note, args.expires, args.expires),
            )
            if cur.rowcount:
                made.append(code)

    if len(made) < args.number:
        print(f"only generated {len(made)} of {args.number} (collisions)", file=sys.stderr)

    seats = len(made) * args.uses
    expiry = f", expiring in {args.expires} days" if args.expires else ""
    print(f"\ncreated {len(made)} code(s), {args.uses} seats each "
          f"= {seats} signups{expiry}\n")
    for code in made:
        print(f"    {code}")
    print()


def cmd_list(conn, args: argparse.Namespace) -> None:
    rows = conn.execute(
        """
        select code, used_count, max_uses, revoked, expires_at, note
          from public.invite_codes
         order by created_at desc, code
        """
    ).fetchall()

    if not rows:
        print("no invite codes yet -- create some with: invite_codes.py new --uses 5")
        return

    print(f"\n  {'CODE':<12} {'USED':>7}  {'LEFT':>5}  {'STATUS':<9} {'NOTE'}")
    print(f"  {'-' * 12} {'-' * 7}  {'-' * 5}  {'-' * 9} {'-' * 20}")

    now = datetime.now(timezone.utc)
    live = spent = 0
    for code, used, cap, revoked, expires, note in rows:
        if revoked:
            status = "revoked"
        elif expires and expires < now:
            status = "expired"
        elif used >= cap:
            status = "full"
        else:
            status = "open"
            live += cap - used
        spent += used
        print(f"  {code:<12} {f'{used}/{cap}':>7}  {cap - used:>5}  "
              f"{status:<9} {note or ''}")

    print(f"\n  {len(rows)} codes | {spent} seats used | {live} still open\n")


def cmd_revoke(conn, args: argparse.Namespace) -> None:
    code = args.code.strip().upper()
    with conn.cursor() as cur:
        cur.execute(
            "update public.invite_codes set revoked = true where code = %s", (code,)
        )
        if not cur.rowcount:
            print(f"no such code: {code}", file=sys.stderr)
            raise SystemExit(1)
    # Revoking does not un-spend seats already taken; people who are already
    # through the gate keep their accounts. It only stops further entry.
    print(f"revoked {code} -- no further signups, existing accounts unaffected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="create invite codes")
    p_new.add_argument("-n", "--number", type=int, default=1,
                       help="how many codes to create (default 1)")
    p_new.add_argument("-u", "--uses", type=int, default=1,
                       help="seats per code (default 1)")
    p_new.add_argument("--note", default=None, help="reminder of who it went to")
    p_new.add_argument("--expires", type=int, default=None,
                       help="days until the code stops working (default never)")
    p_new.set_defaults(func=cmd_new)

    p_list = sub.add_parser("list", help="show every code and its usage")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke", help="stop a code from being used further")
    p_revoke.add_argument("code")
    p_revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    if getattr(args, "uses", 1) is not None and getattr(args, "uses", 1) < 1:
        parser.error("--uses must be at least 1")

    with _connect() as conn:
        args.func(conn, args)


if __name__ == "__main__":
    main()
